# main.py
import os
import re
import base64
import uuid
import string
import logging
import threading
from functools import lru_cache
from typing import List, Dict, Any
from urllib.parse import unquote, urljoin

import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ───────────────────────────── Logging ─────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ───────────────────────── Static mount ────────────────────────────
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
# Serve your generated mp4s
app.mount("/videos", StaticFiles(directory=STATIC_DIR, html=False), name="videos")

# In-memory job store — run a single process/instance
video_jobs: Dict[str, Dict[str, Any]] = {}

# ─────────────────────────── Helpers ───────────────────────────────
def decode_data_uri(s: str) -> bytes:
    """Accepts raw base64 or data URLs; fixes padding and url-encoding."""
    s = (s or "").strip()
    if s.startswith("data:"):
        parts = s.split(",", 1)
        s = parts[1] if len(parts) == 2 else ""
    s = unquote(s)
    s = s.replace("\n", "").replace("\r", "").replace(" ", "")
    s = s.replace("-", "+").replace("_", "/")
    s = re.sub(r"[^A-Za-z0-9+/=]", "", s)
    s += "=" * ((4 - len(s) % 4) % 4)
    return base64.b64decode(s)

def _strip_punct(t: str) -> str:
    return t.translate(str.maketrans("", "", string.punctuation)).lower()

_SIGNASL_BASES = ("https://www.signasl.org/", "https://signasl.org/")

@lru_cache(maxsize=4096)
def _fetch_signasl_urls_for_token(token: str) -> List[str]:
    """
    Try JSON first on each base, then scrape the HTML /sign/<token> page for mp4/webm sources.
    Returns absolute URLs (deduped).
    """
    token = _strip_punct(token or "")
    if not token:
        return []

    found: List[str] = []

    # 1) JSON API (not always available)
    for base in _SIGNASL_BASES:
        try:
            rj = requests.get(urljoin(base, f"api/sign/{token}"), timeout=8)
            if rj.ok:
                data = rj.json()
                if isinstance(data, list):
                    for item in data:
                        u = (item or {}).get("video_url")
                        if u:
                            found.append(u)
        except Exception as e:
            logging.debug("JSON lookup %s failed for %r: %s", base, token, e)

    if found:
        seen, out = set(), []
        for u in found:
            if u not in seen:
                out.append(u); seen.add(u)
        return out

    # 2) HTML scrape for <video>/<source> src or data-src (.mp4 / .webm)
    src_regex = re.compile(r'(?:src|data-src)=["\']([^"\']+?\.(?:mp4|webm))(?:\?[^"\']*)?["\']', re.IGNORECASE)
    for base in _SIGNASL_BASES:
        try:
            rh = requests.get(urljoin(base, f"sign/{token}"), timeout=8)
            if not rh.ok:
                continue
            html = rh.text
            for m in src_regex.findall(html):
                found.append(urljoin(base, m))
        except Exception as e:
            logging.debug("HTML scrape %s failed for %r: %s", base, token, e)

    # de-dupe preserve order
    seen, out = set(), []
    for u in found:
        if u not in seen:
            out.append(u); seen.add(u)
    return out

def translate_text_to_sign(sentence: str) -> List[str]:
    """
    Build a list of clip URLs for the transcript. Word-first, then letters fallback.
    """
    words = _strip_punct(sentence or "").split()
    out: List[str] = []

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

# ─────────────────────────── Schema ────────────────────────────────
class AudioPayload(BaseModel):
    filename: str
    content_base64: str  # data:...;base64,... or raw base64

# ─────────────────────────── Routes ────────────────────────────────
@app.post("/translate_audio/", status_code=200)
async def translate_audio(data: AudioPayload):
    """
    Kick off a job. Body: { filename, content_base64 }
    Returns: { job_id }
    """
    job_id = str(uuid.uuid4())
    video_jobs[job_id] = {"status": "processing", "transcript": ""}

    try:
        audio_bytes = decode_data_uri(data.content_base64)
    except Exception as e:
        video_jobs[job_id] = {"status": "error", "error": f"Invalid base64: {e}"}
        return JSONResponse(status_code=400, content={"status": "error", "error": "Invalid base64"})

    ext = os.path.splitext(data.filename or "")[1].lower()
    if ext not in {".mp3", ".wav", ".m4a", ".aac", ".mp4"}:
        ext = ".mp3"
    temp_audio_path = f"temp_{job_id}{ext}"
    with open(temp_audio_path, "wb") as f:
        f.write(audio_bytes)

    # background worker in same process
    from worker import process_audio_worker  # late import to avoid circular imports
    threading.Thread(
        target=process_audio_worker,
        args=(job_id, temp_audio_path, video_jobs, translate_text_to_sign, STATIC_DIR),
        daemon=True,
    ).start()

    return {"job_id": job_id}

@app.get("/video_status/{job_id}")
def video_status(job_id: str):
    job = video_jobs.get(job_id)
    if not job:
        return {"status": "not_found"}
    if job.get("status") == "ready":
        return {
            "status": "ready",
            "video_url": job.get("video_url"),
            "transcript": job.get("transcript", "")
        }
    if job.get("status") == "error":
        return {"status": "error", "error": job.get("error")}
    return {"status": "processing"}

@app.get("/")
def health():
    return {"status": "ok"}

# ───────────── Optional debug helpers ─────────────
@app.get("/debug_ffmpeg")
def debug_ffmpeg():
    import subprocess, shutil
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return JSONResponse(status_code=500, content={"ok": False, "error": "ffmpeg not found"})
    out = os.path.join(STATIC_DIR, "ffmpeg_test.mp4")
    cmd = [ffmpeg, "-y", "-f", "lavfi", "-i", "color=c=black:s=320x240:d=1",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", out]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return {"ok": True, "url": "/videos/ffmpeg_test.mp4", "size": os.path.getsize(out)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})

@app.get("/debug_aai_key")
def debug_aai_key():
    key = os.getenv("ASSEMBLYAI_API_KEY")
    if not key:
        return {"ok": False, "error": "ASSEMBLYAI_API_KEY not set"}
    r = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        headers={"Authorization": key, "Content-Type": "application/json"},
        json={"audio_url": "https://example.com/does-not-exist.mp3"},
        timeout=10,
    )
    # 400 => key accepted (bad request due to fake URL). 401 => invalid key.
    return {"ok": r.status_code != 401, "status": r.status_code, "body": r.text[:160]}

@app.get("/debug_signasl/{token}")
def debug_signasl(token: str):
    urls = _fetch_signasl_urls_for_token(token)
    return {"token": token, "count": len(urls), "urls": urls[:10]}
