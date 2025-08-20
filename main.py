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
from typing import List, Tuple, Optional

import requests
from fastapi import FastAPI, UploadFile, File, Form
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

# ── FFmpeg for MoviePy (Render-safe) ───────────────────────────────────────────
try:
    import imageio_ffmpeg
    ffbin = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ["IMAGEIO_FFMPEG_EXE"] = ffbin
    os.environ["FFMPEG_BINARY"] = ffbin
    logging.info("🎬 FFmpeg set to %s", ffbin)
except Exception as e:
    logging.warning("FFmpeg setup warning: %s", e)

# ── App / static ───────────────────────────────────────────────────────────────
STATIC_DIR = os.path.join(os.getcwd(), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
app.mount("/videos", StaticFiles(directory=STATIC_DIR, html=False), name="videos")

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

# ── Base64 helper (robust) ─────────────────────────────────────────────────────
def decode_base64_field(field: str) -> bytes:
    """
    Accepts raw base64 or data URLs. Handles url-encoding, urlsafe chars,
    whitespace, and padding.
    """
    s = (field or "").strip()
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
        # fallback without validate for odd inputs
        return base64.b64decode(s)

# ── SignASL helpers ────────────────────────────────────────────────────────────
def _strip_punct(t: str) -> str:
    return t.translate(str.maketrans("", "", string.punctuation)).lower()

@lru_cache(maxsize=4096)
def get_asl_video_url(token: str) -> Optional[str]:
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
        logging.error("ASL lookup failed for '%s': %s", token, e)
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

def build_video_plan(assemblyai_words: List[dict]) -> List[Tuple[str, float]]:
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
                logging.warning("⚠️ Skipping clip %s: %s", url, e)

        if not clips:
            raise RuntimeError("No ASL clips available to merge.")

        final = concatenate_videoclips(clips, method="compose")
        final.write_videofile(
            output_path, codec="libx264", audio=False, fps=24, verbose=False, logger=None
        )
        for c in clips:
            try: c.close()
            except Exception: pass

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("Video file not written or empty.")
        logging.info("✅ Wrote video %s (%d bytes)", output_path, os.path.getsize(output_path))
    finally:
        for p in tmp_files:
            try: os.remove(p)
            except Exception: pass

# ── Pydantic model for JSON route (optional) ───────────────────────────────────
class AudioPayload(BaseModel):
    filename: str
    content_base64: str

# Import worker after helpers so we can pass references
from worker import process_audio_worker  # noqa: E402

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.post("/translate_audio_form/")
async def translate_audio_form(
    file: UploadFile = File(None),
    file_b64: Optional[str] = Form(None),
    filename: Optional[str] = Form(None),
    mime: Optional[str] = Form(None),
):
    """
    Multipart upload:
      - Preferred: file=<binary file>
      - Fallback:  file_b64=<data:...;base64,...> (plus optional filename, mime)
    """
    logging.info("🔥 /translate_audio_form called")
    try:
        job_id = str(uuid.uuid4())
        video_jobs[job_id] = {"status": "processing", "transcript": ""}

        data_bytes: Optional[bytes] = None
        out_name = filename

        if file is not None:
            data_bytes = await file.read()
            out_name = out_name or file.filename
        elif file_b64:
            data_bytes = decode_base64_field(file_b64)
            if not out_name:
                ext = ".mp3"
                if mime and "/" in mime:
                    maybe = "." + mime.split("/", 1)[1]
                    if 1 <= len(maybe) <= 5:
                        ext = maybe
                out_name = f"upload{ext}"

        if not data_bytes:
            return JSONResponse(status_code=422, content={"status": "error", "error": "No file or base64 provided"})

        ext = os.path.splitext(out_name or "")[1].lower() or ".mp3"
        if ext not in {".mp3", ".wav", ".m4a", ".aac", ".mp4"}:
            ext = ".mp3"
        temp_audio_path = f"temp_{job_id}{ext}"

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

@app.post("/translate_audio/")
async def translate_audio(data: AudioPayload):
    """JSON body alternative: { filename, content_base64 }"""
    logging.info("🔥 /translate_audio called")
    try:
        job_id = str(uuid.uuid4())
        video_jobs[job_id] = {"status": "processing", "transcript": ""}
        logging.info("CREATE job %s; known_jobs=%d", job_id, len(video_jobs))

        ext = os.path.splitext(data.filename or "")[1].lower()
        if ext not in {".mp3", ".wav", ".m4a", ".aac", ".mp4"}:
            ext = ".mp3"
        temp_audio_path = f"temp_{job_id}{ext}"

        try:
            audio_bytes = decode_base64_field(data.content_base64)
        except Exception as e:
            logging.error("❌ Base64 decoding failed: %s", e)
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

@app.get("/video_status/{job_id}")
def video_status(job_id: str):
    job = video_jobs.get(job_id)
    if not job:
        return {"status": "not_found"}
    if job.get("status") == "ready":
        url = job.get("video_url") or f"/videos/output_{job_id}.mp4"
        return {"status": "ready", "video_url": url, "transcript": job.get("transcript", "")}
    if job.get("status") == "error":
        return {"status": "error", "error": job.get("error")}
    return {"status": "processing"}

@app.get("/")
def health_check():
    return {"status": "ok"}

# ── Optional debug helpers ─────────────────────────────────────────────────────
@app.post("/echo_form/")
async def echo_form(file: UploadFile = File(None), file_b64: Optional[str] = Form(None)):
    size = None
    if file is not None:
        b = await file.read()
        size = len(b)
    return {
        "has_file": file is not None,
        "file_b64_len": len(file_b64 or ""),
        "file_size": size,
    }

@app.get("/debug_ffmpeg")
def debug_ffmpeg():
    try:
        from moviepy.editor import ColorClip
        out = os.path.join(STATIC_DIR, "ffmpeg_test.mp4")
        clip = ColorClip((320, 240), color=(0, 0, 0), duration=1)
        clip.write_videofile(out, codec="libx264", fps=24, audio=False, verbose=False, logger=None)
        clip.close()
        return {"ok": True, "url": "/videos/ffmpeg_test.mp4", "size": os.path.getsize(out)}
    except Exception as e:
        logging.exception("ffmpeg test failed")
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})
