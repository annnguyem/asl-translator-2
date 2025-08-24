# worker.py
import os
import time
import json
import shutil
import logging
import tempfile
import subprocess

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------- AssemblyAI ----------
def _get_aai_key() -> str:
    key = os.getenv("ASSEMBLYAI_API_KEY")
    if not key:
        raise RuntimeError("ASSEMBLYAI_API_KEY not set")
    return key

def transcribe_with_assemblyai(audio_path: str) -> dict:
    """
    Uploads a local audio file to AssemblyAI and polls for transcript.
    Returns the full transcript JSON (includes 'text' and 'words').
    """
    api_key = _get_aai_key()
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
        raise ValueError(f"Audio file missing or too small: {audio_path}")

    headers = {"authorization": api_key}

    # 1) Upload raw bytes (octet-stream)
    logging.info("⏳ Uploading audio to AssemblyAI (%s bytes)", os.path.getsize(audio_path))
    with open(audio_path, "rb") as f:
        up = requests.post(
            "https://api.assemblyai.com/v2/upload",
            headers={**headers, "content-type": "application/octet-stream"},
            data=f,
            timeout=60,
        )
    up.raise_for_status()
    upload_url = up.json()["upload_url"]
    logging.info("✅ Uploaded. URL: %s", upload_url)

    # 2) Create transcript
    dual_channel = os.getenv("AAI_DUAL_CHANNEL", "false").lower() in ("1", "true", "yes")
    body = {
        "audio_url": upload_url,
        "punctuate": True,
        "format_text": True,
        "speaker_labels": False,
        "dual_channel": dual_channel,
        # Add options as needed…
    }
    tr = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        headers={**headers, "content-type": "application/json"},
        json=body,
        timeout=30,
    )
    tr.raise_for_status()
    tid = tr.json()["id"]
    logging.info("📝 Transcript ID: %s", tid)

    # 3) Poll
    while True:
        r = requests.get(f"https://api.assemblyai.com/v2/transcript/{tid}", headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        status = data.get("status")
        logging.info("🔄 AAI status: %s", status)
        if status == "completed":
            return data
        if status == "error":
            raise RuntimeError(f"AssemblyAI error: {data.get('error')}")
        time.sleep(2)


# ---------- Media fetch/convert (mp4/webm/m3u8) ----------
def _ffmpeg_bin() -> str:
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        # Help MoviePy find it too
        os.environ["IMAGEIO_FFMPEG_EXE"] = ff
        os.environ["FFMPEG_BINARY"] = ff
        return ff
    except Exception:
        return "ffmpeg"  # hope it's in PATH

def _download_clip_to_mp4(url: str) -> str:
    """
    Returns a local **.mp4** path for the given media URL.
    - For HLS (.m3u8): uses ffmpeg to fetch & mux.
    - For .mp4/.webm: downloads, converts webm→mp4 if needed.
    """
    import requests

    ua = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

    if url.lower().endswith(".m3u8"):
        out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        cmd = [
            _ffmpeg_bin(), "-y",
            "-headers", f"User-Agent: {ua}\r\nAccept: */*\r\nReferer: https://www.signasl.org/\r\n",
            "-i", url,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-an",
            out
        ]
        logging.info("📥 ffmpeg HLS → %s", out)
        cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if cp.returncode != 0:
            raise RuntimeError(f"ffmpeg m3u8 fetch failed: {cp.stderr.decode(errors='ignore')[:400]}")
        return out

    # Direct file
    r = requests.get(url, timeout=20, headers={"User-Agent": ua, "Referer": "https://www.signasl.org/"})
    r.raise_for_status()
    lower = url.lower()
    if lower.endswith(".mp4"):
        fn = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        with open(fn, "wb") as f:
            f.write(r.content)
        return fn

    # webm -> mp4
    if lower.endswith(".webm"):
        webm = tempfile.NamedTemporaryFile(delete=False, suffix=".webm").name
        with open(webm, "wb") as f:
            f.write(r.content)
        out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        cmd = [_ffmpeg_bin(), "-y", "-i", webm, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", out]
        logging.info("♻️ webm→mp4 %s", out)
        cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            os.remove(webm)
        except Exception:
            pass
        if cp.returncode != 0:
            raise RuntimeError(f"ffmpeg webm→mp4 failed: {cp.stderr.decode(errors='ignore')[:400]}")
        return out

    # Unknown extension → try ffmpeg directly
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    cmd = [_ffmpeg_bin(), "-y", "-i", url, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", out]
    logging.info("⚙️ ffmpeg generic fetch → %s", out)
    cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if cp.returncode != 0:
        raise RuntimeError(f"ffmpeg fetch failed: {cp.stderr.decode(errors='ignore')[:400]}")
    return out


def generate_merged_video(video_plan, output_path):
    """
    video_plan = [(media_url, duration_seconds), ...]
    Downloads each segment to local mp4 (handling HLS/webm) and concatenates with MoviePy.
    """
    from moviepy.editor import VideoFileClip, concatenate_videoclips

    tmp_files, clips = [], []
    try:
        for url, dur in video_plan:
            try:
                local_mp4 = _download_clip_to_mp4(url)
                tmp_files.append(local_mp4)
                clips.append(VideoFileClip(local_mp4).set_duration(max(float(dur), 0.08)))
            except Exception as e:
                logging.warning("⚠️ skip clip %s: %s", url, e)

        if not clips:
            raise RuntimeError("No ASL clips available to merge.")

        final = concatenate_videoclips(clips, method="compose")
        final.write_videofile(
            output_path, codec="libx264", audio=False, fps=24, verbose=False, logger=None
        )
        for c in clips:
            try:
                c.close()
            except Exception:
                pass

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("Video file not written or empty.")
        logging.info("✅ Wrote video %s (%d bytes)", output_path, os.path.getsize(output_path))
    finally:
        for p in tmp_files:
            try:
                os.remove(p)
            except Exception:
                pass


# ---------- Main worker ----------
def process_audio_worker(job_id: str,
                         audio_path: str,
                         video_jobs: dict,
                         translate_text_to_sign,
                         static_dir: str):
    """
    - Transcribe with AssemblyAI
    - For each word, look up SignASL URLs and allocate durations from AAI timings
    - Merge to /videos/output_<job_id>.mp4 (served by main.py)
    - Update `video_jobs[job_id]`
    """
    try:
        logging.info("🎬 [%s] start", job_id)

        # 1) Transcription
        aai = transcribe_with_assemblyai(audio_path)
        transcript = aai.get("text", "") or ""
        words = aai.get("words", []) or []
        logging.info("🗣️ [%s] transcript len=%d, words=%d", job_id, len(transcript), len(words))

        # 2) Build plan using word timings
                # 2) Build a timed plan
        plan = []
        if words:
            prev_end_ms = words[0].get("start", 0)
            for w in words:
                text = (w.get("text") or "").strip().lower()
                start_ms, end_ms = int(w.get("start", 0)), int(w.get("end", 0))
                dur_s = max((end_ms - start_ms) / 1000.0, 0.12)

                # gap before word
                gap_ms = max(start_ms - prev_end_ms, 0)
                if gap_ms > 0:
                    plan.append(("gap", None, gap_ms / 1000.0))

                urls = translate_text_to_sign(text) or []
                if not urls:
                    # fallback: fingerspell
                    for ch in text:
                        letter_url = translate_text_to_sign(ch) or []
                        if letter_url:
                            plan.append(("clip", letter_url[0], dur_s / max(len(text), 1)))
                    if not urls:  # still nothing
                        plan.append(("gap", None, dur_s))
                else:
                    # normal case
                    if len(urls) == 1:
                        plan.append(("clip", urls[0], dur_s))
                    else:
                        per = max(dur_s / len(urls), 0.08)
                        for u in urls:
                            plan.append(("clip", u, per))

                prev_end_ms = end_ms

        if not plan:
            raise RuntimeError("No ASL clips found even after fallback")

        # 3) Render final video
        out_path = os.path.join(static_dir, f"output_{job_id}.mp4")
        generate_merged_video(plan, out_path)

        rel_url = f"/videos/output_{job_id}.mp4"
        video_jobs[job_id] = {"status": "ready", "video_url": rel_url, "transcript": transcript}
        logging.info("✅ [%s] done, %s", job_id, rel_url)

    except Exception as e:
        logging.error("❌ [%s] failed: %s", job_id, e)
        video_jobs[job_id] = {"status": "error", "error": str(e)}
    finally:
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass
