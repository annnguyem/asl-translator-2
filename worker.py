# worker.py
import os
import time
import json
import shutil
import logging
import tempfile
import subprocess
from typing import Tuple, List

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ============================ Job persistence ============================

def _jobs_dir(static_dir: str) -> str:
    d = os.path.join(static_dir, "jobs")
    os.makedirs(d, exist_ok=True)
    return d

def _write_job(static_dir: str, job_id: str, payload: dict) -> None:
    try:
        p = os.path.join(_jobs_dir(static_dir), f"{job_id}.json")
        with open(p, "w") as f:
            json.dump(payload, f)
    except Exception as e:
        logging.warning("job write failed: %s", e)

# ============================ AssemblyAI ============================

def _get_aai_key() -> str:
    key = os.getenv("ASSEMBLYAI_API_KEY")
    if not key:
        raise RuntimeError("ASSEMBLYAI_API_KEY not set")
    return key

def transcribe_with_assemblyai(audio_path: str) -> dict:
    """
    Uploads local audio to AssemblyAI and polls until complete.
    Returns full transcript JSON (contains 'text' and 'words').
    """
    api_key = _get_aai_key()
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
        raise ValueError(f"Audio file missing or too small: {audio_path}")

    headers = {"authorization": api_key}

    # 1) Upload bytes
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

# ============================ Media fetching / 403 handling ============================

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

def _ffmpeg_bin() -> str:
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    try:
        import imageio_ffmpeg  # type: ignore
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        os.environ["IMAGEIO_FFMPEG_EXE"] = ff
        os.environ["FFMPEG_BINARY"] = ff
        return ff
    except Exception:
        return "ffmpeg"

def _referer_for_url(url: str, token: str) -> Tuple[str, str]:
    """
    Pick the correct referer/origin pair for the CDN domain.
    """
    low = url.lower()
    if "signbsl.com" in low or "media.signbsl.com" in low:
        return (f"https://www.signbsl.com/sign/{token}", "https://www.signbsl.com")
    return (f"https://www.signasl.org/sign/{token}", "https://www.signasl.org")

def _prime_session_for(url: str, token: str) -> Tuple[requests.Session, str, str]:
    """
    Visit the matching sign page to acquire cookies the CDN expects.
    Returns (session, referer_url, cookie_header_string).
    """
    referer, origin = _referer_for_url(url, token or "word")
    s = requests.Session()
    s.headers.update({
        "User-Agent": _UA,
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": origin + "/",
        "Origin": origin,
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    })
    try:
        s.get(referer, timeout=12, allow_redirects=True)
    except Exception:
        pass
    cookie_header = "; ".join([f"{k}={v}" for k, v in s.cookies.get_dict().items()])
    return s, referer, cookie_header

def _download_clip_to_mp4(url: str, token: str = "") -> str:
    """
    Download/convert any media URL to a local .mp4 honoring hotlink protection.
    Requires the word `token` (used to construct a good referer).
    """
    sess, referer, cookie_header = _prime_session_for(url, token or "word")
    origin = referer.split("/sign/")[0]
    headers = {
        "User-Agent": _UA,
        "Referer": referer,
        "Origin": origin,
        "Accept": "*/*",
    }

    lower = url.lower()
    # HLS: use ffmpeg with headers+cookies
    if lower.endswith(".m3u8"):
        out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        ff_headers = (
            f"User-Agent: {_UA}\r\n"
            f"Accept: */*\r\n"
            f"Origin: {origin}\r\n"
            f"Referer: {referer}\r\n"
        )
        if cookie_header:
            ff_headers += f"Cookie: {cookie_header}\r\n"
        cmd = [
            _ffmpeg_bin(), "-y",
            "-headers", ff_headers,
            "-i", url,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-an",
            out
        ]
        logging.info("📥 ffmpeg HLS with headers → %s", out)
        cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if cp.returncode != 0:
            raise RuntimeError(f"ffmpeg m3u8 fetch failed: {cp.stderr.decode(errors='ignore')[:400]}")
        return out

    # mp4/webm via session (cookies + referer)
    r = sess.get(url, headers=headers, timeout=20)
    if not r.ok:
        raise requests.HTTPError(f"{r.status_code} {r.reason} for url: {url}")

    if lower.endswith(".mp4"):
        fn = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        with open(fn, "wb") as f:
            f.write(r.content)
        return fn

    if lower.endswith(".webm"):
        webm = tempfile.NamedTemporaryFile(delete=False, suffix=".webm").name
        with open(webm, "wb") as f:
            f.write(r.content)
        out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        cmd = [_ffmpeg_bin(), "-y", "-i", webm, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", out]
        logging.info("♻️ webm→mp4 %s", out)
        cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try: os.remove(webm)
        except Exception: pass
        if cp.returncode != 0:
            raise RuntimeError(f"ffmpeg webm→mp4 failed: {cp.stderr.decode(errors='ignore')[:400]}")
        return out

    # Unknown → let ffmpeg try with headers
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    ff_headers = (
        f"User-Agent: {_UA}\r\n"
        f"Accept: */*\r\n"
        f"Origin: {origin}\r\n"
        f"Referer: {referer}\r\n"
    )
    if cookie_header:
        ff_headers += f"Cookie: {cookie_header}\r\n"
    cmd = [_ffmpeg_bin(), "-y", "-headers", ff_headers, "-i", url, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", out]
    logging.info("⚙️ ffmpeg generic with headers → %s", out)
    cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if cp.returncode != 0:
        raise RuntimeError(f"ffmpeg fetch failed: {cp.stderr.decode(errors='ignore')[:400]}")
    return out

# ============================ Concatenation ============================

def generate_merged_video(video_plan: List[tuple], output_path: str) -> None:
    """
    video_plan entries can be either:
      (url, duration_s)  OR  (url, duration_s, token_for_referer)
    """
    from moviepy.editor import VideoFileClip, concatenate_videoclips

    tmp_files, clips = [], []
    try:
        for item in video_plan:
            if len(item) == 3:
                url, dur, tok = item
            else:
                url, dur = item
                tok = ""
            try:
                local_mp4 = _download_clip_to_mp4(url, tok)
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
            try: c.close()
            except Exception: pass

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("Video file not written or empty.")
        logging.info("✅ Wrote video %s (%d bytes)", output_path, os.path.getsize(output_path))
    finally:
        for p in tmp_files:
            try: os.remove(p)
            except Exception: pass

# ============================ Worker entry ============================

def process_audio_worker(job_id: str,
                         audio_path: str,
                         video_jobs: dict,
                         translate_text_to_sign,
                         static_dir: str):
    """
    - Transcribe with AssemblyAI
    - For each word, lookup SignASL URLs; allocate durations from AAI timings
    - Merge to /videos/output_<job_id>.mp4 (main.py serves /videos)
    - Persist job state to disk so /video_status reflects changes
    """
    try:
        logging.info("🎬 [%s] start", job_id)

        aai = transcribe_with_assemblyai(audio_path)
        transcript = aai.get("text", "") or ""
        words = aai.get("words", []) or []
        logging.info("🗣️ [%s] transcript len=%d, words=%d", job_id, len(transcript), len(words))

        # Build plan per word, include token for referer
        plan: List[tuple] = []
        for w in words:
            text = (w.get("text") or "").strip()
            try:
                start = int(w.get("start", 0) or 0)
                end   = int(w.get("end",   0) or 0)
            except Exception:
                start, end = 0, 0
            dur_s = max((end - start) / 1000.0, 0.12)
            if not text:
                continue

            try:
                urls = translate_text_to_sign(text) or []
            except Exception as e:
                logging.warning("lookup failed for '%s': %s", text, e)
                urls = []

            logging.info("token '%s' (%.2fs) -> %d url(s)", text, dur_s, len(urls))

            if not urls:
                continue

            if len(urls) == 1:
                plan.append((urls[0], dur_s, text))
            else:
                per = max(dur_s / len(urls), 0.08)
                for u in urls:
                    plan.append((u, per, text))

        if not plan:
            raise RuntimeError("No ASL clips available to merge.")

        out_path = os.path.join(static_dir, f"output_{job_id}.mp4")
        generate_merged_video(plan, out_path)

        rel_url = f"/videos/output_{job_id}.mp4"
        payload = {"status": "ready", "video_url": rel_url, "transcript": transcript}
        video_jobs[job_id] = payload
        _write_job(static_dir, job_id, payload)
        logging.info("✅ [%s] done, %s", job_id, rel_url)

    except Exception as e:
        logging.error("❌ [%s] failed: %s", job_id, e)
        payload = {"status": "error", "error": str(e)}
        video_jobs[job_id] = payload
        _write_job(static_dir, job_id, payload)
    finally:
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass
