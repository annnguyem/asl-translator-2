import os
import sys
import string
import base64
import traceback
import requests
import uuid
import subprocess
import glob
import multiprocessing
import logging
import base64
import re

from fastapi import Query, FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from time import time
from worker import process_audio_worker

# 🔧 Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout
)

multiprocessing.set_start_method("spawn", force=True)

# 🔁 Setup
STATIC_DIR = os.path.join(os.getcwd(), "static")
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
            logging.warning(f"⚠️ Could not delete {f}: {e}")

@app.on_event("startup")
def startup_event():
    logging.info("🚀 Starting up and cleaning temporary files...")
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
        logging.error(f"❌ Failed to get video for '{word}': {e}")
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

def generate_merged_video(video_urls, word_timestamps, output_path):
    import moviepy.editor as mp
    import tempfile

    clips = []

    for idx, (url, timestamp) in enumerate(zip(video_urls, word_timestamps)):
        try:
            # Download video file to temp
            response = requests.get(url)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_vid_file:
                temp_vid_file.write(response.content)
                temp_vid_path = temp_vid_file.name

            # Load clip
            clip = VideoFileClip(temp_vid_path)

            # Calculate desired duration in seconds
            duration_ms = timestamp['end'] - timestamp['start']
            duration = max(duration_ms / 1000.0, 0.1)  # Avoid 0 duration

            # Set exact duration (clip is auto-stretched or trimmed)
            clip = clip.set_duration(duration)

            clips.append(clip)

        except Exception as e:
            print(f"⚠️ Skipping video {url}: {e}")
            continue

    # Combine all clips
    if clips:
        final_video = concatenate_videoclips(clips, method="compose")
        final_video.write_videofile(output_path, codec="libx264", audio=False, fps=24)
        print(f"✅ Video created: {output_path}")
    else:
        print("❌ No clips to merge.")

# 🎙️ API Models
class AudioPayload(BaseModel):
    filename: str
    content_base64: str

# 🔊 POST: Upload audio
@app.post("/translate_audio/")
# Get just the base64 data after "base64,"  
@app.post("/translate_audio/")
async def translate_audio(data: AudioPayload):
    logging.info("🔥 /translate_audio called")

    try:
        job_id = str(uuid.uuid4())
        video_jobs[job_id] = {"status": "processing", "video_urls": [], "transcript": ""}

        temp_audio_path = f"temp_{data.filename}"

        content_base64 = data.content_base64
        if content_base64.startswith("data:"):
            content_base64 = content_base64.split(",")[1]

    # Clean base64 string
    match = re.match(r"^data:audio/\w+;base64,(.*)$", data.content_base64)
    if match:
        content_base64 = match.group(1)
    else:
        content_base64 = data.content_base64.strip()
    
    # Decode base64 safely
    try:
        audio_bytes = base64.b64decode(content_base64)
    except Exception as e:
        logging.error(f"❌ Base64 decoding failed: {e}")
        return {"status": "error", "error": "Invalid base64 audio input"}

# Write to temp file
try:
    with open(temp_audio_path, "wb") as f:
        f.write(audio_bytes)

    file_size = os.path.getsize(temp_audio_path)
    logging.info(f"📦 Saved audio file size: {file_size} bytes")

    if file_size == 0:
        logging.error("❌ Uploaded audio file is 0 bytes.")
        return {"status": "error", "error": "Uploaded audio file is empty"}
except Exception as e:
    logging.error(f"❌ Failed to write audio file: {e}")
    return {"status": "error", "error": "Failed to save audio file"}

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

    except Exception as e:
        logging.error(f"❌ Error in /translate_audio/: {e}")
        return JSONResponse(status_code=500, content={"message": "Internal Server Error"})

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
