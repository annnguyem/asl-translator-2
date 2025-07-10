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

    try:
        clips = []

        logging.info(f"ℹ️ generate_merged_video called with {len(video_urls)} video URLs and {len(word_timestamps)} word timestamps")

        length = min(len(video_urls), len(word_timestamps))

        for i in range(length):
            url = video_urls[i]
            word_info = word_timestamps[i]

            start = word_info.get("start", 0) / 1000.0
            end = word_info.get("end", 0) / 1000.0
            duration = max(end - start, 0.1)

            logging.info(f"🔹 Word {i}: start={start:.3f}s, end={end:.3f}s, duration={duration:.3f}s")

            response = requests.get(url, stream=True)
            if response.status_code != 200:
                raise Exception(f"Failed to download video from {url}")
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp_file.write(response.content)
            tmp_file.close()

            clip = mp.VideoFileClip(tmp_file.name)
            clip_duration = clip.duration
            speed_factor = clip_duration / duration if duration > 0 else 1

            logging.info(f"    Original clip duration: {clip_duration:.3f}s, speed_factor: {speed_factor:.3f}")

            adjusted_clip = clip.fx(mp.vfx.speedx, speed_factor)
            clips.append(adjusted_clip)

        logging.info(f"ℹ️ Concatenating {len(clips)} clips...")
        final_clip = mp.concatenate_videoclips(clips, method="compose")

        logging.info(f"ℹ️ Writing final video to {output_path}...")
        final_clip.write_videofile(output_path, codec="libx264", audio=False, verbose=True)

        for clip in clips:
            clip.close()
            try:
                os.unlink(clip.filename)
            except Exception as e:
                logging.warning(f"⚠️ Could not delete temp clip file {clip.filename}: {e}")

        logging.info(f"✅ Merged video created at {output_path}")

    except Exception as e:
        logging.error(f"❌ Failed to merge videos with timing: {e}")
        raise

# 🎙️ API Models
class AudioPayload(BaseModel):
    filename: str
    content_base64: str

# 🔊 POST: Upload audio
@app.post("/translate_audio/")
async def translate_audio(data: AudioPayload):
    logging.info("🔥 /translate_audio called")

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
