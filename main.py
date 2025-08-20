# main.py
import os
import re
import base64
import uuid
import string
import logging
import threading
from functools import lru_cache
from typing import List, Dict, Any, Optional
from urllib.parse import unquote, urljoin

import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ───────────────────────────── Logging ─────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ───────────────────────── Static mount ────────────────────────────
STATIC_DIR = "static_output"
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# serve final mp4s from here
app.mount("/videos", StaticFiles(directory=STATIC_DIR, html=False), name="videos")

# In-memory job store — run a single process/instance
video_jobs: Dict[str, Dict[str, Any]] = {}

# ─────────────────────────── Helpers ───────────────────────────────
def decode_data_uri(s: str) -> bytes:
    """
    Accepts raw base64 or data URLs. Handles url-encoding, urlsa
