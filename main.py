import os
import sys
import glob
import re
import uuid
import base64
import logging
import tempfile
import string
from functools import lru_cache
from urllib.parse import unquote
from typing import List, Tuple

import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)

# ── FFmpeg for MoviePy (works on Render) ───────────────────────────────────────
try:
    from moviepy.config import change_settings
    import imageio_ffmpeg
    change_settings({"FFMPEG_BINARY": imageio_ffmpeg.get_ffmpeg_exe()})
    logging.info("🎬 FFmpeg configured via imageio-ffmpeg")
except Exception as e:
    logging.warning(f"FFmpeg setup warning: {e}")

# ── App / static ───────────────────────────────────────────────────────────────
STATIC_DIR = os.path.join(os.getcwd(), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# In-memory job store
video_jobs: dict = {}

def clean_temp_files():
    for pattern in (
        "temp_*.mp3", "temp_*.wav", "temp_*.m4a", "temp_*.aac", "temp_*.mp4",
        os.path.join(STATIC_DIR, "output_*.mp4"),
    ):
        for f in glob.glob(pattern):
            try:
                os.remove(f)
            except Exception as e:
                logging.warning(f"⚠️ Could not delete {f}: {e}")

@app.on_event("startup")
def _startup():
    logging.info("🚀 Startup: cleaning temp files")
    clean_temp_files()

# ── Base64 helper ──────────────────────────────────────────────────────────────
def decode_base64_field(field: str) -> bytes:
    """
    Accepts raw base64 or data URLs. Handles url-encoding, urlsafe chars,
    whitespace, and padding.
    """
    s = (field or "").strip()
    logging.info(f"[upload] prefix: {s[:40]!r}")

    if s.startswith("data:"):
        parts = s.split(",", 1)
        s = parts[1] if len(parts) == 2 else ""

    s = unquote(s)
    s = s.replace("\n", "").replace("\r", "").replace(" ", "")
    s = s.replace("-", "+").replace("_", "/")
    s = re.sub(r"[^A-Za-z0-9+/=]", "", s)
    missing = (-len(s)) % 4
    if missing:
        s += "=" * missing

    try:
        return base64.b64decode(s, validate=True)
    except Exception:
        return base64.b64decode(s)

# ── SignASL helpers ────────────────────────────────────────────────────────────
def _strip_punct(t: str) -> str:
    return t.translate(str.maketrans("", "", string.punctuation)).lower()
