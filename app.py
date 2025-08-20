# app.py
import os, io, string, base64, uuid, traceback, requests, threading
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

STATIC_DIR = "static_output"
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
# ✅ serve files we create
app.mount("/videos", StaticFiles(directory=STATIC_DIR, html=False), name="videos")

from worker import process_audio_worker  # after STATIC_DIR is defined

# ---- helpers ----
def strip_punctuation(text): return text.translate(str.maketrans("", "", string.punctuation)).lower()

def get_asl_video_url(token: str):
    try:
        r = requests.get(f"https://signasl.org/api/sign/{token}", timeout=8)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            return data[0].get("video_url")
    except Exception as e:
        print(f"ASL lookup failed for '{token}': {e}")
    return None

def translate_text_to_sign(sentence: str):
    words = strip_punctuation(sentence or "").split()
    urls = []
    for w in words:
        u = get_asl_video_url(w)
        if u: urls.append(u); continue
        for ch in w:
            cu = get_asl_video_url(ch)
            if cu: urls.append(cu)
    return urls

def decode_data_uri(s: str) -> bytes:
    s = (s or "").strip()
    if s.startswith("data:"):
        s = s.split(",", 1)[1] if "," in s else ""
    s = s.replace("\n","").replace("\r","").replace(" ","")
    # handle missing padding
    s += "=" * ((4 - len(s) % 4) % 4)
    return base64.b64decode(s)

# ---- routes ----
class AudioPayload(BaseModel):
    filename: str
    content_base64: str

# in-memory store (single process/instance)
video_jobs = {}

@app.post("/translate_audio/", status_code=200)
async def translate_audio(data: AudioPayload):
    job_id = str(uuid.uuid4())
    video_jobs[job_id] = {"status": "processing", "transcript": ""}

    audio_bytes = decode_data_uri(data.content_base64)
    ext = os.path.splitext(data.filename or "")[1].lower() or ".mp3"
    temp_audio_path = f"temp_{job_id}{ext}"
    with open(temp_audio_path, "wb") as f:
        f.write(audio_bytes)

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
        # ✅ return the single merged file URL
        return {
            "status": "ready",
            "video_url": job.get("video_url"),
            "transcript": job.get("transcript", "")
        }
    if job.get("status") == "error":
        return {"status": "error", "error": job.get("error")}
    return {"status": "processing"}

@app.get("/")
def health_check():
    return {"status": "ok"}

# optional: quick ffmpeg sanity
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
