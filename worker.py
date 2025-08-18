import os, time, logging, requests, traceback
from dotenv import load_dotenv
import urllib.parse

_orig_send = requests.Session.send
def _logged_send(self, request, **kwargs):
    host = urllib.parse.urlparse(request.url).netloc
    logging.info("[egress] %s %s", request.method, request.url)
    return _orig_send(self, request, **kwargs)
requests.Session.send = _logged_send

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
load_dotenv()  # loads .env if present (e.g., created by Actions or on your server)

def _get_api_key() -> str:
    key = os.getenv("ASSEMBLYAI_API_KEY")  # do NOT hardcode
    if not key:
        raise RuntimeError("ASSEMBLYAI_API_KEY not set")

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

def transcribe_with_assemblyai(audio_path: str) -> dict:
    api_key = _get_api_key()

    if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
        raise ValueError(f"Audio file missing or too small: {audio_path}")

    # 1) Upload
    logging.info(f"⏳ Uploading {audio_path} ({os.path.getsize(audio_path)} bytes)")
    with open(audio_path, "rb") as f:
        up = requests.post(
            "https://api.assemblyai.com/v2/upload",
            headers={"authorization": api_key, "content-type": "application/octet-stream"},
            data=f,
            timeout=60,
        )
    up.raise_for_status()
    upload_url = up.json()["upload_url"]
    logging.info(f"✅ Uploaded. URL: {upload_url}")

    # 2) Submit transcript
    dual_channel = os.getenv("AAI_DUAL_CHANNEL", "false").lower() == "true"  # set true for stereo call audio
    body = {
        "audio_url": upload_url,
        "punctuate": True,
        "format_text": True,
        "speaker_labels": False,
        "dual_channel": dual_channel,
    }
    tr = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        headers={"authorization": api_key, "content-type": "application/json"},
        json=body,
        timeout=30,
    )
    tr.raise_for_status()
    tid = tr.json()["id"]
    logging.info(f"📝 Transcript ID: {tid}")

    # 3) Poll
    while True:
        r = requests.get(f"https://api.assemblyai.com/v2/transcript/{tid}", headers={"authorization": api_key}, timeout=30)
        r.raise_for_status()
        data = r.json()
        status = data.get("status")
        logging.info(f"🔄 Transcription status: {status}")
        if status == "completed":
            return data
        if status == "error":
            raise RuntimeError(f"AssemblyAI error: {data.get('error')}")
        time.sleep(2)

def process_audio_worker(job_id: str, audio_path: str, video_jobs: dict,
                         lookup_sign_urls_for_word, build_video_plan, generate_merged_video, static_dir: str):
    try:
        logging.info(f"🎬 [{job_id}] Start")
        data = transcribe_with_assemblyai(audio_path)
        transcript = data.get("text", "") or ""
        words = data.get("words", []) or []

        # Build plan (if your lookup function is needed, pass it through)
        plan = build_video_plan(words)  # or build_video_plan(words, lookup_sign_urls_for_word)

        # Render final mp4 to the same static_dir your web app serves
        out_path = os.path.join(static_dir, f"output_{job_id}.mp4")
        generate_merged_video(plan, out_path)

        # 🔑 Tell the frontend where to load it
        rel_url = f"/videos/output_{job_id}.mp4"   # must match the mount in main.py
        video_jobs[job_id] = {"status": "ready", "transcript": transcript, "video_url": rel_url}
        logging.info(f"✅ [{job_id}] Done, video at {out_path}")

    except Exception as e:
        logging.error(f"❌ [{job_id}] Failed: {e}")
        logging.debug("Traceback:\n" + traceback.format_exc())
        video_jobs[job_id] = {"status": "error", "error": str(e)}
    finally:
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass
