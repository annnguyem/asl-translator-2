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
ASL_VIDEO_DIR = "videos_database"
STATIC_DIR = "static_output"
GDRIVE_FILE_ID = "1T2A6VjUbmozDtoXecq9-e2e3wSjY7czu"
ZIP_FILENAME = "asl_videos.zip"

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

def get_asl_video_path(word):
    file_name = f"{word}.mp4"
    file_path = os.path.join(ASL_VIDEO_DIR, file_name)
    return file_path if os.path.isfile(file_path) else None

def translate_text_to_sign(sentence):
    clean_sentence = strip_punctuation(sentence)
    words = clean_sentence.split()

    asl_video_paths = []
    for word in words:
        video_path = get_asl_video_path(word)
        if video_path:
            asl_video_paths.append(video_path)
        else:
            for letter in word:
                letter_path = get_asl_video_path(letter)
                if letter_path:
                    asl_video_paths.append(letter_path)
    return asl_video_paths

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
    video_jobs[job_id] = {"status": "processing", "path": None}

    temp_audio_path = f"temp_{data.filename}"
    audio_bytes = base64.b64decode(data.content_base64)
    with open(temp_audio_path, "wb") as f:
        f.write(audio_bytes)

    print(f"📥 Received file {data.filename}, base64 length: {len(data.content_base64)}")

    import threading
    threading.Thread(
        target=process_audio_worker,
        args=(job_id, temp_audio_path, video_jobs, STATIC_DIR, translate_text_to_sign, generate_merged_video),
        daemon=True
    ).start()

    return {"job_id": job_id}

@app.get("/video_status/{job_id}")
def video_status(job_id: str):
    video_path = os.path.join(STATIC_DIR, f"output_{job_id}.mp4")
    done_path = video_path.replace(".mp4", ".done")

    if os.path.exists(done_path):
        url = f"/static/output_{job_id}.mp4?t={int(time())}"
        return {
            "status": "ready",
            "video_url": url,
            "transcript": video_jobs[job_id].get("transcript", "")
        }
    elif os.path.exists(video_path):
        return {"status": "processing"}
    else:
        return {"status": "not_found"}

@app.get("/translated_text")
def translated_text(sentence: str = Query(...)):
    clean_sentence = strip_punctuation(sentence)
    words = clean_sentence.split()

    translated = []
    for word in words:
        if get_asl_video_path(word):
            translated.append(word)
        else:
            translated += [letter for letter in word if get_asl_video_path(letter)]

    return {"translated": translated}

@app.get("/")
def health_check():
    print("✅ Health check OK")
    return {"status": "ok"}