import os
import io
import string
import base64
import traceback
import requests
import uuid
import multiprocessing

from fastapi import Query, FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from time import time, sleep
from worker import process_audio_worker

multiprocessing.set_start_method("spawn", force=True)

STATIC_DIR = "static_output"
os.makedirs(STATIC_DIR, exist_ok=True)

# 🚀 FastAPI app setup
app = FastAPI()
ASSEMBLYAI_API_KEY = 'dbb3ea03ff1a43468beef535573eb703'

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

# 🔤 Utilities
def strip_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation)).lower()

def get_asl_video_url(word):
    try:
        response = requests.get(f"https://signasl.org/api/sign/{word}")
        response.raise_for_status()
        results = response.json()
        if results and isinstance(results, list):
            return results[0].get("video_url")
    except Exception as e:
        print(f"❌ Failed to get video for '{word}': {e}")
    return None

def translate_text_to_sign(sentence):
    clean_sentence = strip_punctuation(sentence)
    words = clean_sentence.split()

    urls = []
    for word in words:
        url = get_asl_video_url(word)
        if url:
            urls.append(url)
        else:
            for letter in word:
                letter_url = get_asl_video_url(letter)
                if letter_url:
                    urls.append(letter_url)
    return urls

# 📅 Base64 audio endpoint
class AudioPayload(BaseModel):
    filename: str
    content_base64: str

@app.post("/translate_audio/", status_code=200)
async def translate_audio(data: AudioPayload):
    job_id = str(uuid.uuid4())
    video_jobs[job_id] = {"status": "processing", "video_urls": [], "transcript": ""}

    temp_audio_path = f"temp_{data.filename}"
    audio_bytes = base64.b64decode(data.content_base64)
    with open(temp_audio_path, "wb") as f:
        f.write(audio_bytes)

    print(f"📥 Received audio file: {data.filename}")

    import threading
    threading.Thread(
        target=process_audio_worker,
        args=(job_id, temp_audio_path, video_jobs, translate_text_to_sign),
        daemon=True
    ).start()

    return {"job_id": job_id}

@app.get("/video_status/{job_id}")
def video_status(job_id: str):
    if job_id in video_jobs:
        job = video_jobs[job_id]
        return {
            "status": job["status"],
            "video_urls": job.get("video_urls", []),
            "transcript": job.get("transcript", "")
        }
    return {"status": "not_found"}

@app.get("/")
def health_check():
    print("✅ Health check OK")
    return {"status": "ok"}
