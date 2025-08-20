import os, base64, uuid, string, re, logging, requests, threading
from functools import lru_cache
from urllib.parse import unquote
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── static mount ───────────────────────────────────────────────────────────────
STATIC_DIR = "static_output"
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
app.mount("/videos", StaticFiles(directory=STATIC_DIR, html=False), name="videos")

# in-memory job store (use one instance/worker)
video_jobs: dict[str, dict] = {}

# ── helpers ───────────────────────────────────────────────────────────────────
def decode_data_uri(s: str) -> bytes:
    """Accept data URLs or raw base64; fix padding/encoding quirks."""
    s = (s or "").strip()
    if s.startswith("data:"):
        s = s.split(",", 1)[1] if "," in s else ""
    s = unquote(s)
    s = s.replace("\n", "").replace("\r", "").replace(" ", "")
    s = s.replace("-", "+").replace("_", "/")
    s = re.sub(r"[^A-Za-z0-9+/=]", "", s)
    s += "=" * ((4 - len(s) % 4) % 4)
    return base64.b64decode(s)

def _strip_punct(t: str) -> str:
    return t.translate(str.maketrans("", "", string.punctuation)).lower()

@lru_cache(maxsize=4096)
def get_asl_video_url(token: str) -> str | None:
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
        logging.warning("ASL lookup failed for '%s': %s", token, e)
    return None

def translate_text_to_sign(sentence: str) -> list[str]:
    """Return a list of source clip URLs for the transcript text."""
    words = _strip_punct(sentence or "").split()
    urls: list[str] = []
    for w in words:
        u = get_asl_video_url(w)
        if u:
            urls.append(u)
            continue
        for ch in w:
            cu = get_asl_video_url(ch)
            if cu:
                urls.append(cu)
    return urls

# ── schema ────────────────────────────────────────────────────────────────────
class AudioPayload(BaseModel):
    filename: str
    content_base64: str  # can be data:...;base64,...

# ── routes ────────────────────────────────────────────────────────────────────
@app.post("/translate_audio/", status_code=200)
async def translate_audio(data: AudioPayload):
    """Kick off a job. Body: { filename, content_base64 }."""
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

    from worker import process_audio_worker  # late import, no circulars
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
            "video_url": job.get("video_url"),  # singular field your poller expects
            "transcript": job.get("transcript", "")
        }
    if job.get("status") == "error":
        return {"status": "error", "error": job.get("error")}
    return {"status": "processing"}

@app.get("/")
def health():
    return {"status": "ok"}

# optional: ffmpeg sanity check
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

# optional: AssemblyAI key sanity
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
    # 400 = key accepted (bad request due to fake URL). 401 = invalid key.
    return {"ok": r.status_code != 401, "status": r.status_code, "body": r.text[:160]}
