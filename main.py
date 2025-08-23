import os
import io
import string
import base64
import uuid
import multiprocessing
import threading

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from worker import process_audio_worker

multiprocessing.set_start_method("spawn", force=True)

STATIC_DIR = "static_output"
os.makedirs(STATIC_DIR, exist_ok=True)

# Serve the generated videos under /videos/...
app = FastAPI()
app.mount("/videos", StaticFiles(directory=STATIC_DIR), name="videos")

# ── CORS ────────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job status storage
manager = multiprocessing.Manager()
video_jobs = manager.dict()

# If you keep an inline key, also mirror it to env so worker sees it.
ASSEMBLYAI_API_KEY = 'dbb3ea03ff1a43468beef535573eb703'
os.environ["ASSEMBLYAI_API_KEY"] = ASSEMBLYAI_API_KEY

# ── Utilities ──────────────────────────────────────────────────────────────────
def strip_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation)).lower()

import requests

def get_asl_video_url(token: str):
    try:
        r = requests.get(f"https://signasl.org/api/sign/{token}", timeout=15)
        r.raise_for_status()
        results = r.json()
        if results and isinstance(results, list):
            return results[0].get("video_url")
    except Exception as e:
        print(f"❌ Failed to get video for '{token}': {e}")
    return None

def translate_text_to_sign(sentence: str):
    """
    Returns a list of URLs for a sentence, falling back to fingerspelling
    per letter when a whole-word sign isn't found.
    """
    clean = strip_punctuation(sentence)
    words = clean.split()

    urls = []
    for w in words:
        url = get_asl_video_url(w)
        if url:
            urls.append(url)
        else:
            for ch in w:
                letter_url = get_asl_video_url(ch)
                if letter_url:
                    urls.append(letter_url)
    return urls

# ── API ────────────────────────────────────────────────────────────────────────
class AudioPayload(BaseModel):
    filename: str
    content_base64: str

@app.post("/translate_audio/", status_code=200)
async def translate_audio(data: AudioPayload):
    job_id = str(uuid.uuid4())
    video_jobs[job_id] = {"status": "processing", "video_url": "", "transcript": ""}

    temp_audio_path = f"temp_{data.filename}"
    audio_bytes = base64.b64decode(data.content_base64)
    with open(temp_audio_path, "wb") as f:
        f.write(audio_bytes)

    print(f"📥 Received audio file: {data.filename}")

    threading.Thread(
        target=process_audio_worker,
        args=(job_id, temp_audio_path, video_jobs, translate_text_to_sign, STATIC_DIR),
        daemon=True
    ).start()

    return {"job_id": job_id}

@app.get("/video_status/{job_id}")
def video_status(job_id: str):
    job = video_jobs.get(job_id)
    if job:
        return job
    return {"status": "not_found"}

@app.get("/")
def health_check():
    print("✅ Health check OK")
    return {"status": "ok"}
