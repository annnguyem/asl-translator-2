import os, time, requests, tempfile, subprocess, shutil, traceback

def _aai_key():
    # Prefer environment variable; NEVER hardcode in production
    key = os.getenv("ASSEMBLYAI_API_KEY")
    if not key:
        raise RuntimeError("ASSEMBLYAI_API_KEY not set")
    return key

def transcribe_with_assemblyai(audio_path: str) -> str:
    key = _aai_key()
    # 1) upload (raw bytes, not multipart)
    with open(audio_path, "rb") as f:
        up = requests.post(
            "https://api.assemblyai.com/v2/upload",
            headers={"Authorization": key},
            data=f,
            timeout=60,
        )
    if not up.ok:
        raise RuntimeError(f"Upload failed: {up.status_code} {up.text[:200]}")
    upload_url = up.json()["upload_url"]
    # 2) request transcript
    tr = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        headers={"Authorization": key, "Content-Type": "application/json"},
        json={"audio_url": upload_url, "punctuate": True, "format_text": True},
        timeout=30,
    )
    if not tr.ok:
        raise RuntimeError(f"Transcript create failed: {tr.status_code} {tr.text[:200]}")
    tid = tr.json()["id"]
    # 3) poll
    while True:
        r = requests.get(f"https://api.assemblyai.com/v2/transcript/{tid}",
                         headers={"Authorization": key}, timeout=30)
        r.raise_for_status()
        d = r.json()
        if d.get("status") == "completed":
            return d.get("text", "")
        if d.get("status") == "error":
            raise RuntimeError(f"AssemblyAI error: {d.get('error')}")
        time.sleep(2)

def generate_merged_video(urls, output_path):
    if not urls:
        raise RuntimeError("No ASL clips available to merge.")
    # find ffmpeg
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    # download all clips
    tmp_dir = tempfile.mkdtemp(prefix="asl_")
    listfile = os.path.join(tmp_dir, "list.txt")
    paths = []
    for i, u in enumerate(urls):
        try:
            r = requests.get(u, timeout=12)
            r.raise_for_status()
            p = os.path.join(tmp_dir, f"clip_{i}.mp4")
            with open(p, "wb") as f: f.write(r.content)
            paths.append(p)
        except Exception as e:
            print(f"skip {u}: {e}")
    if not paths:
        raise RuntimeError("All ASL clip downloads failed.")

    # build concat list
    with open(listfile, "w") as f:
        for p in paths:
            f.write(f"file '{p}'\n")

    # robust concat (re-encode to avoid codec/size mismatch)
    cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", listfile,
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "24", "-an", output_path]
    subprocess.run(cmd, check=True, capture_output=True)

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("Video file not written or empty.")

def process_audio_worker(job_id, audio_path, video_jobs, translate_text_to_sign, static_dir):
    try:
        print(f"[{job_id}] transcribing…")
        transcript = transcribe_with_assemblyai(audio_path)
        print(f"[{job_id}] transcript: {transcript!r}")

        urls = translate_text_to_sign(transcript)
        print(f"[{job_id}] {len(urls)} ASL clips")

        out_path = os.path.join(static_dir, f"output_{job_id}.mp4")
        generate_merged_video(urls, out_path)

        # ✅ expose the ONE merged file for the frontend
        video_jobs[job_id] = {
            "status": "ready",
            "video_url": f"/videos/output_{job_id}.mp4",
            "transcript": transcript
        }
        print(f"[{job_id}] done: {out_path}")
    except Exception as e:
        traceback.print_exc()
        video_jobs[job_id] = {"status": "error", "error": str(e)}
    finally:
        try:
            if os.path.exists(audio_path): os.remove(audio_path)
        except Exception:
            pass
