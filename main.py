import os
import string
import base64
import traceback
import requests
import uuid
import subprocess
import glob
import multiprocessing

from fastapi import Query, FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from time import time
from worker import process_audio_worker

multiprocessing.set_start_method("spawn", force=True)

# 🔁 Setup
STATIC_DIR = "static_output"
os.makedirs(STATIC_DIR, exist_ok=True)

# 🚀 FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# In-memory job store
manager = multiprocessing.Manager()
video_jobs = manager.dict()

# 🧹 Clean up old temp/output files
def clean_temp_files():
    for f in glob.glob("temp_*.mp3") + glob.glob(os.path.join(STATIC_DIR, "output_*.mp4")):
        try:
            os.remove(f)
        except Exception as e:
            print(f"⚠️ Could not delete {f}: {e}")

@app.on_event("startup")
def startup_event():
    clean_temp_files()

# 🔤 Helpers
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

def generate_merged_video(video_urls, output_path):
    try:
        input_txt = "input.txt"
        with open(input_txt, "w") as f:
            for url in video_urls:
                filename = url.split("/")[-1]
                local_path = os.path.join(STATIC_DIR, filename)
                with open(local_path, "wb") as vid_file:
                    vid_file.write(requests.get(url).content)
                f.write(f"file '{local_path}'\n")

        cmd = ["ffmpeg", "-f", "concat", "-safe", "0", "-i", input_txt, "-c", "copy", output_path]
        subprocess.run(cmd, check=True)
        os.remove(input_txt)

    except Exception as e:
        print(f"❌ Failed to merge videos: {e}")
        raise

# 🎙️ API Models
class AudioPayload(BaseModel):
    filename: str
    content_base64: str

# 🔊 POST: Upload audio
@app.post("/translate_audio/")
async def translate_audio(data: AudioPayload):
    job_id = str(uuid.uuid4())
    video_jobs[job_id] = {"status": "processing", "video_urls": [], "transcript": ""}

    temp_audio_path = f"temp_{data.filename}"
    with open(temp_audio_path, "wb") as f:
        f.write(base64.b64decode(data.content_base64))

    import threading
    threading.Thread(
        target=process_audio_worker,
        args=(
            job_id,
            temp_audio_path,
            video_jobs,
            translate_text_to_sign,
            generate_merged_video,
            STATIC_DIR
        ),
        daemon=True
    ).start()

    return {"job_id": job_id}

# 🎞️ GET: Poll job status
@app.get("/video_status/{job_id}")
def video_status(job_id: str):
    if job_id not in video_jobs:
        return {"status": "not_found"}

    job = video_jobs[job_id]
    if job["status"] == "ready":
        return {
            "status": "ready",
            "video_url": f"/static/output_{job_id}.mp4",
            "transcript": job.get("transcript", "")
        }
    elif job["status"] == "error":
        return {"status": "error"}
    return {"status": "processing"}

# ✅ GET: Health check
@app.get("/")
def health_check():
    return {"status": "ok"}
