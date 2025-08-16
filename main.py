import os, sys, string, base64, traceback, requests, uuid, glob, logging, re, tempfile, threading
from functools import lru_cache
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from worker import process_audio_worker

# 🔧 Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)

# 🔁 Setup
STATIC_DIR = os.path.join(os.getcwd(), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Shared job store
video_jobs = {}

def clean_temp_files():
    for f in glob.glob("temp_*.mp3") + glob.glob("temp_*.wav") + glob.glob(os.path.join(STATIC_DIR, "output_*.mp4")):
        try:
            os.remove(f)
        except Exception as e:
            logging.warning(f"⚠️ Could not delete {f}: {e}")

@app.on_event("startup")
def startup_event():
    logging.info("🚀 Starting up and cleaning temporary files...")
    clean_temp_files()

# ---------- ASL lookup & video planning ----------

def _strip_punct(t: str) -> str:
    return t.translate(str.maketrans("", "", string.punctuation)).lower()

@lru_cache(maxsize=4096)
def get_asl_video_url(token: str) -> str | None:
    token = _strip_punct(token)
    if not token:
        return None
    try:
        r = requests.get(f"https://signasl.org/api/sign/{token}", timeout=5)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            return data[0].get("video_url")
    except Exception as e:
        logging.error(f"ASL lookup failed for '{token}': {e}")
    return None

def lookup_sign_urls_for_word(word: str) -> list[str]:
    """Return 1+ URLs for a word; fall back to finger-spelling."""
    urls = []
    word_clean = _strip_punct(word)
    if not word_clean:
        return urls
    u = get_asl_video_url(word_clean)
    if u:
        return [u]
    for ch in word_clean:  # finger-spell
        u = get_asl_video_url(ch)
        if u:
            urls.append(u)
    return urls

def build_video_plan(assemblyai_words: list[dict]) -> list[tuple[str, float]]:
    """
    Turn AssemblyAI words (with ms start/end) into a list of (video_url, duration_seconds),
    splitting a word's duration across its letter clips if needed.
    """
    plan: list[tuple[str, float]] = []
    for w in assemblyai_words:
        text = w.get("text", "")
        start = int(w.get("start", 0) or 0)
        end = int(w.get("end", 0) or 0)
        dur_ms = max(end - start, 100)  # avoid 0
        urls = lookup_sign_urls_for_word(text)
        if not urls:
            continue
        if len(urls) == 1:
            plan.append((urls[0], dur_ms / 1000.0))
        else:
            per = (dur_ms / 1000.0) / len(urls)
            for u in urls:
                plan.append((u, max(per, 0.08)))  # floor tiny clips
    return plan

def generate_merged_video(video_plan: list[tuple[str, float]], output_path: str) -> None:
    from moviepy.editor import VideoFileClip, concatenate_videoclips
    tmp_files: list[str] = []
    clips = []
    try:
        for url, dur in video_plan:
            try:
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tf:
                    tf.write(r.content)
                    tmp_files.append(tf.name)
                clip = VideoFileClip(tf.name).set_duration(dur)
                clips.append(clip)
            except Exception as e:
                logging.warning(f"⚠️ Skipping clip {url}: {e}")
        if not clips:
            raise RuntimeError("No ASL clips available to merge.")
        final = concatenate_videoclips(clips, method="compose")
        final.write_videofile(output_path, codec="libx264", audio=False, fps=24)
    finally:
        for c in clips:
            try: c.close()
            except: pass
        for p in tmp_files:
            try: os.remove(p)
            except: pass

# ---------- API ----------

class AudioPayload(BaseModel):
    filename: str
    content_base64: str

@app.post("/translate_audio/")
async def translate_audio(data: AudioPayload):
    logging.info("🔥 /translate_audio called")
    try:
        job_id = str(uuid.uuid4())
        video_jobs[job_id] = {"status": "processing", "transcript": ""}

        # sanitize extension and write to a job-scoped temp file
        ext = os.path.splitext(data.filename)[1].lower()
        if ext not in {".mp3", ".wav", ".m4a", ".aac", ".mp4"}:
            ext = ".mp3"
        temp_audio_path = f"temp_{job_id}{ext}"

        # strip data URI prefix if present
        b64 = data.content_base64 or ""
        if b64.startswith("data:"):
            parts = b64.split(",", 1)
            b64 = parts[1] if len(parts) == 2 else ""

        try:
            audio_bytes = base64.b64decode(b64)
        except Exception as e:
            return JSONResponse(status_code=400, content={"status": "error", "error": f"Invalid base64 audio: {e}"})
        with open(temp_audio_path, "wb") as f:
            f.write(audio_bytes)
        if os.path.getsize(temp_audio_path) == 0:
            return JSONResponse(status_code=400, content={"status": "error", "error": "Uploaded audio is empty"})

        threading.Thread(
            target=process_audio_worker,
            args=(
                job_id,
                temp_audio_path,
                video_jobs,
                lookup_sign_urls_for_word,  # per-word lookup
                build_video_plan,           # assemble (url, duration) plan
                generate_merged_video,
                STATIC_DIR,
            ),
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
        return {"status": "ready", "video_url": f"/static/output_{job_id}.mp4", "transcript": job.get("transcript", "")}
    if job.get("status") == "error":
        return {"status": "error", "error": job.get("error")}
    return {"status": "processing"}

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/debug_audio/")
async def debug_audio(data: AudioPayload):
    try:
        decoded = base64.b64decode(data.content_base64 or "")
        return {"filename": data.filename, "base64_length": len(data.content_base64 or ""), "decoded_length": len(decoded)}
    except Exception as e:
        return {"error": str(e)}
