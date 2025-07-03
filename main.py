import os
import io
import string
import base64
import zipfile
import traceback
import glob
import gdown
import uuid
import multiprocessing
import moviepy
import subprocess

from moviepy.video.io.VideoFileClip import VideoFileClip
from pydantic import BaseModel
from fastapi import Query
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from time import time, sleep
from worker import process_audio_worker

multiprocessing.set_start_method("spawn", force=True)

# 🔠 Constants
STATIC_DIR = "static_output"

# In-memory job status storage
manager = multiprocessing.Manager()
video_jobs = manager.dict()

# ✅ Make sure static dir exists BEFORE mounting
os.makedirs(STATIC_DIR, exist_ok=True)

# 🚀 FastAPI app setup
app = FastAPI()
ASSEMBLYAI_API_KEY = '2b791d89824a4d5d8eeb7e310aa6542f'

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ⬇️ Download + Extract Videos from Google Drive
def download_and_extract_videos():
    if os.path.exists(ASL_VIDEO_DIR):
        print("📂 Video database already exists. Skipping download.")
        return

    print("⬇️ Downloading ASL video database from Google Drive...")
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    gdown.download(url, ZIP_FILENAME, quiet=False)

    with zipfile.ZipFile(ZIP_FILENAME, 'r') as zip_ref:
        zip_ref.extractall()

    os.remove(ZIP_FILENAME)
    print("✅ Extracted video database to:", ASL_VIDEO_DIR)

def clean_temp_files():
    print("🪩 Cleaning up old files...")
    for pattern in ["temp_*.mp3"]:
        for file in glob.glob(pattern):
            try:
                os.remove(file)
                print(f"🗑️ Removed {file}")
            except Exception as e:
                print(f"⚠️ Failed to remove {file}: {e}")
    for file in glob.glob(os.path.join(STATIC_DIR, "output_*.mp4")):
        try:
            os.remove(file)
            print(f"🗑️ Removed {file}")
        except Exception as e:
            print(f"⚠️ Failed to remove {file}: {e}")

@app.on_event("startup")
def startup_event():
    clean_temp_files()
    download_and_extract_videos()

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

def generate_merged_video(video_paths, output_path):
    try:
        if not video_paths:
            raise ValueError("No video paths provided")

        input_args = []
        filter_parts = []
        for idx, path in enumerate(video_paths):
            input_args += ["-i", path]
            filter_parts.append(f"[{idx}:v]scale=640:360,fps=30[v{idx}]")

        filter_complex = "; ".join(filter_parts)
        concat_inputs = "".join(f"[v{idx}]" for idx in range(len(video_paths)))
        filter_complex += f"; {concat_inputs}concat=n={len(video_paths)}:v=1:a=0[outv]"

        cmd = [
            "ffmpeg",
            "-y",
            *input_args,
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            output_path
        ]

        subprocess.run(cmd, check=True)
        print(f"✅ Merged video created at {output_path}")

    except subprocess.CalledProcessError as e:
        print(f"❌ ffmpeg failed: {e}")
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise

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
