import os
import sys
import glob
import re
import uuid
import base64
import logging
import tempfile
import string
from functools import lru_cache
from urllib.parse import unquote
from typing import List, Tuple

import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)

# ── FFmpeg for MoviePy (works on Render) ───────────────────────────────────────
try:
    from moviepy.config import change_settings
    import imageio_ffmpeg
    change_settings({"FFMPEG_BINARY": imageio_ffmpeg.get_ffmpeg_exe()})
    logging.info("🎬 FFmpeg configured via imageio-ffmpeg")
except Exception as e:
    logging.warning(f"FFmpeg setup warning: {e}")

# ── App / static ───────────────────────────────────────────────────────────────
STATIC_DIR = os.path.join(os.getcwd(), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# In-memory job store
video_jobs: dict = {}

def clean_temp_files():
    for pattern in (
        "temp_*.mp3", "temp_*.wav", "temp_*.m4a", "temp_*.aac", "temp_*.mp4",
        os.path.join(STATIC_DIR, "output_*.mp4"),
    ):
        for f in glob.glob(pattern):
            try:
                os.remove(f)
            except Exception as e:
                logging.warning(f"⚠️ Could not delete {f}: {e}")

@app.on_event("startup")
def _startup():
    logging.info("🚀 Startup: cleaning temp files")
    clean_temp_files()

# ── Base64 helper ──────────────────────────────────────────────────────────────
def decode_base64_field(field: str) -> bytes:
    """
    Accepts raw base64 or data URLs. Handles url-encoding, urlsafe chars,
    whitespace, and padding.
    """
    s = (field or "").strip()
    logging.info(f"[upload] prefix: {s[:40]!r}")

    if s.startswith("data:"):
        parts = s.split(",", 1)
        s = parts[1] if len(parts) == 2 else ""

    s = unquote(s)
    s = s.replace("\n", "").replace("\r", "").replace(" ", "")
    s = s.replace("-", "+").replace("_", "/")
    s = re.sub(r"[^A-Za-z0-9+/=]", "", s)
    missing = (-len(s)) % 4
    if missing:
        s += "=" * missing

    try:
        return base64.b64decode(s, validate=True)
    except Exception:
        return base64.b64decode(s)

# ── SignASL helpers ────────────────────────────────────────────────────────────
def _strip_punct(t: str) -> str:
    return t.translate(str.maketrans("", "", string.punctuation)).lower()

@lru_cache(maxsize=4096)
def get_asl_video_url(token: str):
    token = _strip_punct(token or "")
    if not token:
        return None
    try:
        r = requests.get(f"https://signasl.org/api/sign/{token}", timeout=8)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            return data[0].get("video_url")
    except Exception as e:
        logging.error(f"ASL lookup failed for '{token}': {e}")
    return None

def lookup_sign_urls_for_word(word: str) -> List[str]:
    urls: List[str] = []
    w = _strip_punct(word or "")
    if not w:
        return urls
    whole = get_asl_video_url(w)
    if whole:
        return [whole]
    for ch in w:
        u = get_asl_video_url(ch)
        if u:
            urls.append(u)
    return urls

def build_video_plan(assemblyai_words: list) -> List[Tuple[str, float]]:
    """
    Input: [{"text","start","end"}, ...]
    Output: [(url, duration_s), ...]
    """
    plan: List[Tuple[str, float]] = []
    for w in assemblyai_words or []:
        text = w.get("text", "")
        start = int(w.get("start", 0) or 0)
        end = int(w.get("end", 0) or 0)
        dur_ms = max(end - start, 100)
        urls = lookup_sign_urls_for_word(text)
        if not urls:
            continue
        if len(urls) == 1:
            plan.append((urls[0], dur_ms / 1000.0))
        else:
            per = (dur_ms / 1000.0) / len(urls)
            per = max(per, 0.08)
            for u in urls:
                plan.append((u, per))
    return plan

def generate_merged_video(video_plan: List[Tuple[str, float]], output_path: str) -> None:
    from moviepy.editor import VideoFileClip, concatenate_videoclips
    tmp_files, clips = [], []
    try:
        for url, dur in video_plan:
            try:
                r = requests.get(url, timeout=12)
                r.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tf:
                    tf.write(r.content)
                    tmp_files.append(tf.name)
                clips.append(VideoFileClip(tf.name).set_duration(max(float(dur), 0.08)))
            except Exception as e:
                logging.warning(f"⚠️ Skipping clip {url}: {e}")

        if not clips:
            raise RuntimeError("No ASL clips available to merge.")

        final = concatenate_videoclips(clips, method="compose")
        final.write_videofile(output_path, codec="libx264", audio=False, fps=24, verbose=False, logger=None)
        for c in clips:
            try: c.close()
            except Exception: pass

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("Video file not written or empty.")
        logging.info(f"✅ Wrote video {output_path} ({os.path.getsize(output_path)} bytes)")
    finally:
        for p in tmp_files:
            try: os.remove(p)
            except Exception: pass

# ── Pydantic model for JSON route ──────────────────────────────────────────────
class AudioPayload(BaseModel):
    filename: str
    content_base64: str

# Import worker after helpers (so we can pass references)
from worker import process_audio_worker  # noqa: E402

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.post("/translate_audio/")
async def translate_audio(data: AudioPayload):
    logging.info("🔥 /translate_audio called")
    try:
        job_id = str(uuid.uuid4())
        video_jobs[job_id] = {"status": "processing", "transcript": ""}

        ext = os.path.splitext(data.filename or "")[1].lower()
        if ext not in {".mp3", ".wav", ".m4a", ".aac", ".mp4"}:
            ext = ".mp3"
        temp_audio_path = f"temp_{job_id}{ext}"

        try:
            audio_bytes = decode_base64_field(data.content_base64)
        except Exception as e:
            logging.error(f"❌ Base64 decoding failed: {e}")
            return JSONResponse(status_code=400, content={"status": "error", "error": f"Invalid base64 audio: {e}"})

        with open(temp_audio_path, "wb") as f:
            f.write(audio_bytes)
        if os.path.getsize(temp_audio_path) == 0:
            return JSONResponse(status_code=400, content={"status": "error", "error": "Uploaded audio file is empty"})

        import threading
        threading.Thread(
            target=process_audio_worker,
            args=(job_id, temp_audio_path, video_jobs, lookup_sign_urls_for_word, build_video_plan, generate_merged_video, STATIC_DIR),
            daemon=True,
        ).start()

        return {"job_id": job_id}
    except Exception as e:
        logging.exception("❌ Error in /translate_audio")
        return JSONResponse(status_code=500, content={"status": "error", "error": str(e)})

@app.post("/translate_audio_form/")
async def translate_audio_form(file: UploadFile = File(...)):
    logging.info("🔥 /translate_audio_form called")
    try:
        job_id = str(uuid.uuid4())
        video_jobs[job_id] = {"status": "processing", "transcript": ""}

        ext = os.path.splitext(file.filename or "")[1].lower() or ".mp3"
        if ext not in {".mp3", ".wav", ".m4a", ".aac", ".mp4"}:
            ext = ".mp3"
        temp_audio_path = f"temp_{job_id}{ext}"

        data_bytes = await file.read()
        if not data_bytes:
            return JSONResponse(status_code=400, content={"status": "error", "error": "Uploaded file is empty"})

        with open(temp_audio_path, "wb") as f:
            f.write(data_bytes)

        import threading
        threading.Thread(
            target=process_audio_worker,
            args=(job_id, temp_audio_path, video_jobs, lookup_sign_urls_for_word, build_video_plan, generate_merged_video, STATIC_DIR),
            daemon=True,
        ).start()

        return {"job_id": job_id}
    except Exception as e:
        logging.exception("❌ Error in /translate_audio_form")
        return JSONResponse(status_code=500, content={"status": "error", "error": str(e)})

@app.get("/video_status/{job_id}")
def video_status(job_id: str):
    job = video_jobs.get(job_id)
    if not job:
        return {"status": "not_found"}
    if job.get("status") == "ready":
        return {
            "status": "ready",
            "video_url": f"/static/output_{job_id}.mp4",
            "transcript": job.get("transcript", ""),
        }
    if job.get("status") == "error":
        return {"status": "error", "error": job.get("error")}
    return {"status": "processing"}

@app.get("/")
def health_check():
    return {"status": "ok"}

# Optional debug endpoints
@app.post("/echo/")
async def echo(payload: dict):
    return {
        "keys": list(payload.keys()),
        "filename": payload.get("filename"),
        "base64_prefix": (payload.get("content_base64") or "")[:30],
        "base64_len": len(payload.get("content_base64") or ""),
    }
