import os
import re
import base64
import uuid
import string
import logging
import threading
from urllib.parse import unquote, urljoin

import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# -------------------- Logging --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -------------------- Static mount --------------------
STATIC_DIR = "static_output"
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/videos", StaticFiles(directory=STATIC_DIR, html=False), name="videos")

# In-memory job store (run ONE instance/worker)
video_jobs = {}

# -------------------- Helpers --------------------
def decode_data_uri(s):
    # Accept data URLs or raw base64; normalize and fix padding
    s = (s or "").strip()
    if s.startswith("data:"):
        parts = s.split(",", 1)
        s = parts[1] if len(parts) == 2 else ""
    s = unquote(s)
    s = s.replace("\n", "").replace("\r", "").replace(" ", "")
    s = s.replace("-", "+").replace("_", "/")
    s = re.sub(r"[^A-Za-z0-9+/=]", "", s)
    pad = (4 - (len(s) % 4)) % 4
    if pad:
        s += "=" * pad
    return base64.b64decode(s)

def _strip_punct(t):
    return t.translate(str.maketrans("", "", string.punctuation)).lower()

# Try both bases (some sites use www, some not)
_SIGNASL_BASES = ("https://www.signasl.org/", "https://signasl.org/")

def _browser_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.signasl.org/",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    })
    return s

def _fetch_signasl_urls_for_token(token):
    token = _strip_punct(token or "")
    if not token:
        return []

    sess = _browser_session()
    found = []

    # 1) JSON API (if exposed)
    for base in _SIGNASL_BASES:
        url = urljoin(base, "api/sign/" + token)
        try:
            rj = sess.get(url, timeout=8, allow_redirects=True)
            if rj.ok:
                data = rj.json()
                if isinstance(data, list):
                    for item in data:
                        u = (item or {}).get("video_url")
                        if u:
                            found.append(u)
        except Exception as e:
            logging.debug("JSON %s failed (%s): %s", url, token, e)

    # 2) HTML scrape
    attr_re = re.compile(r'(?:src|data-src|srcset)=["\']([^"\']+?\.(?:mp4|webm)(?:\?[^"\']*)?)["\']', re.IGNORECASE)
    abs_re = re.compile(r'https?://[^\s"\'<>]+?\.(?:mp4|webm)\b', re.IGNORECASE)

    for base in _SIGNASL_BASES:
        page = urljoin(base, "sign/" + token)
        try:
            rh = sess.get(page, timeout=8, allow_redirects=True)
            if not rh.ok:
                continue
            html = rh.text

            for m in attr_re.findall(html):
                found.append(urljoin(base, m))
            for m in abs_re.findall(html):
                found.append(m)
        except Exception as e:
            logging.debug("HTML %s failed (%s): %s", page, token, e)

    # de-dupe, preserve order
    seen, out = set(), []
    for u in found:
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out

def translate_text_to_sign(sentence):
    words = _strip_punct(sentence or "").split()
    out = []
    for w in words:
        hits = _fetch_signasl_urls_for_token(w)
        if hits:
            out.extend(hits)
            continue
        # fallback: letters
        for ch in w:
            hits_ch = _fetch_signasl_urls_for_token(ch)
            if hits_ch:
                out.extend(hits_ch)
    return out

# -------------------- Schema --------------------
class AudioPayload(BaseModel):
    filename: str
    content_base64: str  # data:...;base64,... or raw base64

# -------------------- Routes --------------------
@app.post("/translate_audio/", status_code=200)
async def translate_audio(data: AudioPayload):
    job_id = str(uuid.uuid4())
    video_jobs[job_id] = {"status": "processing", "transcript": ""}
