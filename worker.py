import os, time, logging, requests, traceback

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def _get_api_key() -> str:
    key = os.getenv("dbb3ea03ff1a43468beef535573eb703", "").strip()
    if not key:
        raise RuntimeError("ASSEMBLYAI_API_KEY not set")
    return key

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

        # plan & render
        plan = build_video_plan(words)
        out_path = os.path.join(static_dir, f"output_{job_id}.mp4")
        generate_merged_video(plan, out_path)

        video_jobs[job_id] = {"status": "ready", "transcript": transcript}
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
