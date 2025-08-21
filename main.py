import os
import re
import json
import base64
import uuid
import string
import logging
import threading
from typing import Optional
from urllib.parse import unquote, urljoin

import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# -------------------- Logging --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -------------------- Static / Jobs --------------------
STATIC_DIR = "static_output"
os.makedirs(STATIC_DIR, exist_ok=True)

JOBS_DIR = os.path.join(STATIC_DIR, "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

def _job_path(job_id: str) -> str:
    return os.path.join(JOBS_DIR, f"{job_id}.json")

def write_job(job_id: str, payload: dict) -> None:
    try:
        with open(_job_path(job_id), "w") as f:
            json.dump(payload, f)
    except Exception:
        pass

def read_job(job_id: str) -> Optional[dict]:
    p = _job_path(job_id)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None

# -------------------- FastAPI --------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Serve generated videos from /videos/...
app.mount("/videos", StaticFiles(directory=STATIC_DIR, html=False), name="videos")

# In-memory snapshot; disk is source of truth
video_jobs: dict[str, dict] = {}

# -------------------- Helpers --------------------
def decode_data_uri(s: str) -> bytes:
    """
    Accepts raw base64 or data URLs; normalizes and fixes padding.
    """
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

def _strip_punct(t: str) -> str:
    return (t or "").translate(str.maketrans("", "", string.punctuation)).lower().strip()

# -------------------- Minimal SignASL scraper (HTTP only) --------------------
_SIGNASL_BASES = ("https://www.signasl.org/", "https://signasl.org/")

def _browser_session() -> requests.Session:
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

def _fetch_signasl_urls_http(token: str) -> list[str]:
    token = _strip_punct(token)
    if not token:
        return []
    sess = _browser_session()
    found: list[str] = []

    # JSON API (if available)
    for base in _SIGNASL_BASES:
        api = urljoin(base, f"api/sign/{token}")
        try:
            rj = sess.get(api, timeout=8, allow_redirects=True)
            if rj.ok:
                data = rj.json()
                if isinstance(data, list):
                    for item in data:
                        u = (item or {}).get("video_url")
                        if u:
                            found.append(u)
        except Exception:
            pass

    # HTML scrape (mp4/webm/m3u8)
    attr_re = re.compile(
        r'(?:src|data-src|srcset|data-video|data-hls)=["\']([^"\']+?\.(?:mp4|webm|m3u8)(?:\?[^"\']*)?)["\']',
        re.IGNORECASE,
    )
    abs_re = re.compile(r'https?://[^\s"\'<>]+?\.(?:mp4|webm|m3u8)\b', re.IGNORECASE)

    for base in _SIGNASL_BASES:
        page = urljoin(base, f"sign/{token}")
        try:
            rh = sess.get(page, timeout=12, allow_redirects=True)
            if not rh.ok:
                continue
            html = rh.text
            for m in attr_re.findall(html):
                found.append(urljoin(base, m))
            for m in abs_re.findall(html):
                found.append(m)
        except Exception:
            pass

    # de-dupe, preserve order
    seen, out = set(), []
    for u in found:
        if u not in seen:
            out.append(u); seen.add(u)
    return out

def lookup_sign_urls_for_word(word: str) -> list[str]:
    urls = _fetch_signasl_urls_http(word)
    if urls:
        return urls[:2]  # short per-word
    # fallback to fingerspelling a few letters
    out: list[str] = []
    for ch in _strip_punct(word):
        u = _fetch_signasl_urls_http(ch)
        if u:
            out.append(u[0])
        if len(out) >= 6:
            break
    return out

def translate_text_to_sign(sentence: str) -> list[str]:
    tokens = _strip_punct(sentence).split()
    urls: list[str] = []
    for t in tokens:
        urls += lookup_sign_urls_for_word(t)
    return urls

# -------------------- Schemas --------------------
class AudioPayload(BaseModel):
    filename: Optional[str] = None
    content_base64: Optional[str] = None  # data URL or raw base64

# -------------------- Routes --------------------
@app.post("/translate_audio", status_code=200)
@app.post("/translate_audio/", status_code=200)
async def translate_audio(data: AudioPayload | None = None):
    """
    Accepts base64 (or data URL) audio and kicks off a background job that:
    - transcribes via AssemblyAI (done in worker)
    - fetches SignASL clips per word (HTTP scraper above)
    - concatenates to a single MP4 in /videos/output_<job_id>.mp4
    Always returns {"job_id": "..."} so the UI can poll.
    """
    job_id = str(uuid.uuid4())

    # Initialize job state so polling can find it immediately
    init = {"status": "processing", "transcript": ""}
    video_jobs[job_id] = init
    write_job(job_id, init)

    # Validate base64 presence
    if not data or not data.content_base64:
        err = {"status": "error", "error": "content_base64 is required"}
        video_jobs[job_id] = err
        write_job(job_id, err)
        return {"job_id": job_id, "status": "error", "reason": "missing content_base64"}

    # Decode (handles data:...;base64,... and raw base64)
    try:
        audio_bytes = decode_data_uri(data.content_base64)
    except Exception as e:
        err = {"status": "error", "error": f"Invalid base64: {e}"}
        video_jobs[job_id] = err
        write_job(job_id, err)
        return {"job_id": job_id, "status": "error", "reason": "invalid base64"}

    # Choose extension and write temp file
    ext = os.path.splitext((data.filename or "").strip())[1].lower()
    if ext not in {".mp3", ".wav", ".m4a", ".aac", ".mp4"}:
        ext = ".mp3"
    temp_audio_path = f"temp_{job_id}{ext}"
    with open(temp_audio_path, "wb") as f:
        f.write(audio_bytes)

    # Start the background worker
    from worker import process_audio_worker
    threading.Thread(
        target=process_audio_worker,
        args=(job_id, temp_audio_path, video_jobs, translate_text_to_sign, STATIC_DIR),
        daemon=True,
    ).start()

    return {"job_id": job_id}

@app.get("/video_status/{job_id}")
def video_status(job_id: str):
    """
    Returns: {"status": processing|ready|error|not_found, ...}
    """
    job = video_jobs.get(job_id) or read_job(job_id)
    if not job:
        # If the file exists, recover as ready
        out = os.path.join(STATIC_DIR, f"output_{job_id}.mp4")
        if os.path.exists(out):
            payload = {"status": "ready", "video_url": f"/videos/output_{job_id}.mp4", "transcript": ""}
            video_jobs[job_id] = payload
            write_job(job_id, payload)
            return payload
        return {"status": "not_found"}

    st = job.get("status")
    if st == "ready":
        return {
            "status": "ready",
            "video_url": job.get("video_url") or f"/videos/output_{job_id}.mp4",
            "transcript": job.get("transcript", ""),
        }
    if st == "error":
        return {"status": "error", "error": job.get("error")}
    return {"status": "processing"}

@app.get("/")
def health():
    return {"status": "ok"}

# -------------------- Debug / Diagnostics --------------------
@app.get("/whoami")
def whoami():
    import socket
    return {"host": socket.gethostname(), "pid": os.getpid()}

@app.get("/debug_jobs")
def debug_jobs():
    try:
        ids = [f[:-5] for f in os.listdir(JOBS_DIR) if f.endswith(".json")]
    except Exception:
        ids = []
    return {
        "mem_count": len(video_jobs),
        "disk_count": len(ids),
        "mem_ids": list(video_jobs.keys()),
        "disk_ids": ids
    }

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
        headers={"authorization": key, "content-type": "application/json"},
        json={"audio_url": "https://example.com/does-not-exist.mp3"},
        timeout=10,
    )
    # 401 → bad key; 400 → key ok but bad audio URL (expected)
    return {"ok": r.status_code != 401, "status": r.status_code, "body": r.text[:160]}

@app.get("/debug_signasl3/{token}")
def debug_signasl3(token: str):
    urls = _fetch_signasl_urls_http(token)
    return {"token": token, "count": len(urls), "urls": urls[:5]}

# -------------------- Startup housekeeping --------------------
@app.on_event("startup")
def _startup_cleanup():
    # Prune old temp files on boot
    try:
        for name in os.listdir("."):
            if name.startswith("temp_") and (name.endswith(".mp3") or name.endswith(".wav")
                                             or name.endswith(".m4a") or name.endswith(".aac")
                                             or name.endswith(".mp4")):
                try:
                    os.remove(name)
                except Exception:
                    pass
    except Exception:
        pass
