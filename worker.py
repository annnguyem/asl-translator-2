# worker.py
import os
import time
import json
import shutil
import tempfile
import subprocess
import traceback
from typing import List, Dict, Any

import requests


# ----------------------------- Env / tools --------------------------------
def _aai_key() -> str:
    key = os.getenv("ASSEMBLYAI_API_KEY")
    if not key:
        raise RuntimeError("ASSEMBLYAI_API_KEY not set")
    return key


def _ffmpeg_path() -> str:
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise RuntimeError("ffmpeg not found (install system ffmpeg or add `imageio-ffmpeg`)")

# mimic browser for direct downloads
def _browser_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.signasl.org/",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    })
    return s


# ----------------------------- AssemblyAI ---------------------------------
def transcribe_with_assemblyai(audio_path: str) -> dict:
    key = _aai_key()

    # --- 1) Upload raw bytes ---
    with open(audio_path, "rb") as f:
        up = requests.post(
            "https://api.assemblyai.com/v2/upload",
            headers={"authorization": key, "content-type": "application/octet-stream"},
            data=f,
            timeout=60,
        )
    if not up.ok:
        raise RuntimeError(f"Upload failed: {up.status_code} {up.text[:300]}")
    upload_url = up.json()["upload_url"]

    # --- 2) Create transcript (minimal schema first) ---
    body = {
        "audio_url": upload_url,   # REQUIRED
        # Add extras later once working:
        # "punctuate": True,
        # "format_text": True,
        # "dual_channel": os.getenv("AAI_DUAL_CHANNEL", "false").lower() in ("1","true","yes"),
    }
    tr = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        headers={"authorization": key, "content-type": "application/json"},
        json=body,
        timeout=30,
    )
    if not tr.ok:
        # Surface the exact server message so you can see what's wrong
        raise RuntimeError(f"Transcript create failed: {tr.status_code} {tr.text}")

    tid = tr.json()["id"]

    # --- 3) Poll until complete ---
    while True:
        r = requests.get(
            f"https://api.assemblyai.com/v2/transcript/{tid}",
            headers={"authorization": key},
            timeout=30,
        )
        r.raise_for_status()
        d = r.json()
        st = d.get("status")
        if st == "completed":
            return {"text": d.get("text", "") or "", "words": d.get("words", []) or []}
        if st == "error":
            raise RuntimeError(f"AssemblyAI error: {d.get('error')}")
        time.sleep(2)

# ----------------------------- Video utils --------------------------------
def _download_media(url: str) -> str:
    """
    Download a single clip to a temp file.
    - For .m3u8 (HLS), use ffmpeg to remux/encode into a temp .mp4
    - For .mp4 / .webm, download bytes directly
    Returns local file path.
    """
    ffmpeg = _ffmpeg_path()
    lower = url.lower()

    if ".m3u8" in lower:
        out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        out.close()
        hdr = (
            "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36\r\n"
            "Accept: */*\r\n"
            "Accept-Language: en-US,en;q=0.9\r\n"
            "Referer: https://www.signasl.org/\r\n"
            "Cache-Control: no-cache\r\n"
            "Pragma: no-cache\r\n"
        )
        cmd = [
            ffmpeg, "-y",
            "-headers", hdr,
            "-protocol_whitelist", "file,http,https,tcp,tls,crypto",
            "-i", url,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "24",
            "-an",
            out.name,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return out.name

    # direct download for mp4/webm
    sess = _browser_session()
    suffix = ".mp4" if lower.endswith(".mp4") else ".webm" if lower.endswith(".webm") else ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with sess.get(url, timeout=20, stream=True) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_content(1024 * 64):
            if chunk:
                tmp.write(chunk)
    tmp.close()
    return tmp.name


def _concat_with_filter(inputs: List[str], output_path: str) -> None:
    """
    Robust concatenation using ffmpeg concat filter.
    - Normalizes fps, pixel format and size.
    - Handles mixed mp4/webm inputs.
    """
    if not inputs:
        raise RuntimeError("No input files to concat")

    ffmpeg = _ffmpeg_path()

    # Build filter graph:
    #   [0:v]fps=24,format=yuv420p,scale=640:-2,setsar=1[v0]; ... ; [v0][v1]... concat=n=N:v=1:a=0[vout]
    filters = []
    refs = []
    for idx in range(len(inputs)):
        filters.append(f"[{idx}:v]fps=24,format=yuv420p,scale=640:-2,setsar=1[v{idx}]")
        refs.append(f"[v{idx}]")
    filter_complex = ";".join(filters) + f";{''.join(refs)}concat=n={len(inputs)}:v=1:a=0[vout]"

    cmd = [
        ffmpeg, "-y",
        *sum([["-i", p] for p in inputs], []),
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-r", "24",
        "-movflags", "+faststart",
        "-an",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("Video file not written or empty.")


def _placeholder_video(output_path: str, text: str = "") -> None:
    """
    Generate a tiny black MP4 with optional centered text. Used only if all clip
    downloads fail, to keep the UI flow from hanging.
    """
    ffmpeg = _ffmpeg_path()
    safe_text = (text or "No ASL clips found").replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")
    cmd = [
        ffmpeg, "-y",
        "-f", "lavfi", "-i", "color=c=black:s=640x360:d=3",
        "-vf", f"drawtext=text='{safe_text}':fontcolor=white:fontsize=28:x=(w-text_w)/2:y=(h-text_h)/2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "24", "-an",
        output_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except Exception:
        # Fallback without drawtext
        subprocess.run([
            ffmpeg, "-y",
            "-f", "lavfi", "-i", "color=c=black:s=640x360:d=3",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "24", "-an",
            output_path
        ], check=True, capture_output=True)


def _build_video_plan(words: List[Dict[str, Any]], lookup_urls) -> List[Dict[str, Any]]:
    """
    Build a plan array like: [{"url": "...", "dur": 1.1}, ...]
    Uses AssemblyAI word timings to proportion durations across 1..N clips/word.
    """
    plan: List[Dict[str, Any]] = []
    for w in words or []:
        token = str(w.get("text", "")).strip()
        if not token:
            continue
        start = int(w.get("start") or 0)
        end = int(w.get("end") or 0)
        dur_s = max((end - start) / 1000.0, 0.35)  # min per-word visibility

        urls = lookup_urls(token) or []
        if not urls:
            continue

        # share time across multiple clips for the same token
        per = max(dur_s / len(urls), 0.25)
        for u in urls:
            plan.append({"url": u, "dur": per})

    return plan


def generate_merged_video(plan: List[Dict[str, Any]], output_path: str) -> None:
    """
    Download each planned clip and concat.
    """
    tmp_files: List[str] = []
    try:
        for item in plan:
            url = item["url"]
            dur = float(item["dur"])
            try:
                local = _download_media(url)  # mp4/webm/m3u8 -> local file
                tmp_files.append(local)

                # Optionally trim to target duration using ffmpeg (faster than reloading in MoviePy)
                # Here we keep it simple and trim at concat stage via filter fps/scale; most clips are short.
                # If precise trims are required, you can add per-clip trim here.

            except Exception as e:
                print(f"[download] skip {url}: {e}")

        if not tmp_files:
            raise RuntimeError("All ASL clip downloads failed.")

        _concat_with_filter(tmp_files, output_path)

    finally:
        for p in tmp_files:
            try:
                os.remove(p)
            except Exception:
                pass


# ----------------------------- Worker entry -------------------------------
def process_audio_worker(
    job_id: str,
    audio_path: str,
    video_jobs: dict,
    translate_text_to_sign,  # callable: sentence -> list[str] (not used directly here)
    static_dir: str
) -> None:
    """
    Background job runner.
    - Transcribes audio
    - Builds a per-word clip plan using the *word* timestamps and a word->URLs lookup
      provided by main.py (e.g., one that uses Playwright to read signasl.org)
    - Concats clips to /videos/output_<job_id>.mp4
    - Updates video_jobs[job_id]
    """
    try:
        print(f"[{job_id}] transcribing…")
        data = transcribe_with_assemblyai(audio_path)
        transcript = data.get("text", "")
        words = data.get("words", []) or []
        print(f"[{job_id}] transcript: {transcript!r} (words={len(words)})")

        # main.py should expose a function: lookup_sign_urls_for_word(word) -> list[str]
        # for compatibility with your previous wiring, derive it from translate_text_to_sign:
        def lookup_sign_urls_for_word(w: str) -> List[str]:
            # translate_text_to_sign(sentence) returns a list for the whole sentence;
            # for per-word, call with the word only.
            try:
                urls = translate_text_to_sign(w) or []
                # Limit to prevent very long merges
                return urls[:2]
            except Exception:
                return []

        plan = _build_video_plan(words, lookup_sign_urls_for_word)

        out_path = os.path.join(static_dir, f"output_{job_id}.mp4")
        if plan:
            try:
                generate_merged_video(plan, out_path)
            except Exception as merge_err:
                print(f"[{job_id}] merge failed, generating placeholder: {merge_err}")
                _placeholder_video(out_path, transcript)
        else:
            # No clips at all → placeholder so the UI completes
            _placeholder_video(out_path, transcript)

        video_jobs[job_id] = {
            "status": "ready",
            "video_url": f"/videos/output_{job_id}.mp4",
            "transcript": transcript,
        }
        print(f"[{job_id}] done: {out_path}")

    except Exception as e:
        traceback.print_exc()
        video_jobs[job_id] = {"status": "error", "error": str(e)}

    finally:
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass
