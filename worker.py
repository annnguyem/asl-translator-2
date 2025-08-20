# worker.py
import os
import time
import shutil
import tempfile
import subprocess
import traceback
from typing import List

import requests


# ----------------------------- helpers ---------------------------------
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
        raise RuntimeError(
            "ffmpeg not found (install system ffmpeg or add `imageio-ffmpeg`)"
        )


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


# -------------------------- AssemblyAI ---------------------------------
def transcribe_with_assemblyai(audio_path: str) -> str:
    """
    Uploads audio and polls until completed. Returns plain transcript text.
    """
    key = _aai_key()

    # 1) Upload (stream body)
    with open(audio_path, "rb") as f:
        up = requests.post(
            "https://api.assemblyai.com/v2/upload",
            headers={"Authorization": key},
            data=f,
            timeout=60,
        )
    if not up.ok:
        raise RuntimeError(f"Upload failed: {up.status_code} {up.text[:300]}")
    upload_url = up.json()["upload_url"]

    # 2) Create transcript
    tr = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        headers={"Authorization": key, "Content-Type": "application/json"},
        json={"audio_url": upload_url, "punctuate": True, "format_text": True},
        timeout=30,
    )
    if not tr.ok:
        raise RuntimeError(f"Transcript create failed: {tr.status_code} {tr.text[:300]}")
    tid = tr.json()["id"]

    # 3) Poll
    while True:
        r = requests.get(
            f"https://api.assemblyai.com/v2/transcript/{tid}",
            headers={"Authorization": key},
            timeout=30,
        )
        r.raise_for_status()
        d = r.json()
        st = d.get("status")
        if st == "completed":
            return d.get("text", "") or ""
        if st == "error":
            raise RuntimeError(f"AssemblyAI error: {d.get('error')}")
        time.sleep(2)
    # ---------------------------- Video ------------------------------------
    attr_re = re.compile(
        r'(?:src|data-src|srcset|data-video|data-hls)=["\']([^"\']+?\.(?:mp4|webm|m3u8)(?:\?[^"\']*)?)["\']',
        re.IGNORECASE,
    )
    abs_re = re.compile(
        r'https?://[^\s"\'<>]+?\.(?:mp4|webm|m3u8)\b',
        re.IGNORECASE,
    )

def _concat_with_filter(inputs: List[str], output_path: str):
    """
    Use ffmpeg concat filter to robustly join heterogeneous inputs (mp4/webm,
    varying sizes/codecs) into a single H.264 MP4.
    """
    if not inputs:
        raise RuntimeError("No input files to concat")

    ffmpeg = _ffmpeg_path()

    # Build filter graph:
    #   [0:v]fps=24,format=yuv420p,scale=640:-2,setsar=1[v0];
    #   [1:v]fps=24,format=yuv420p,scale=640:-2,setsar=1[v1];
    #   [v0][v1]concat=n=2:v=1:a=0[vout]
    filters = []
    refs = []
    for idx in range(len(inputs)):
        filters.append(f"[{idx}:v]fps=24,format=yuv420p,scale=640:-2,setsar=1[v{idx}]")
        refs.append(f"[v{idx}]")
    filter_complex = ";".join(filters) + f";{''.join(refs)}concat=n={len(inputs)}:v=1:a=0[vout]"

    cmd = [
        ffmpeg, "-y",
        *sum([["-i", p] for p in inputs], []),      # -i file1 -i file2 ...
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


def _placeholder_video(output_path: str, text: str = ""):
    """
    Generate a short black video. Try drawtext; if unavailable, fall back to plain color.
    """
    ffmpeg = _ffmpeg_path()
    # sanitize text for drawtext
    safe_text = (text or "No ASL clips found").replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")
    # Try with drawtext
    cmd1 = [
        ffmpeg, "-y",
        "-f", "lavfi", "-i", "color=c=black:s=640x360:d=3",
        "-vf", f"drawtext=text='{safe_text}':fontcolor=white:fontsize=24:x=(w-text_w)/2:y=(h-text_h)/2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "24", "-an",
        output_path
    ]
    try:
        subprocess.run(cmd1, check=True, capture_output=True)
        return
    except Exception:
        # Fallback: color only (no drawtext)
        cmd2 = [
            ffmpeg, "-y",
            "-f", "lavfi", "-i", "color=c=black:s=640x360:d=3",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "24", "-an",
            output_path
        ]
        subprocess.run(cmd2, check=True, capture_output=True)


def generate_merged_video(urls: List[str], output_path: str):
    """
    Download clips and concatenate into a single MP4.
    Accepts mixed .mp4/.webm; limits to a sane number of clips to avoid very long videos.
    """
    # Limit to avoid extremely long merges; adjust as needed
    urls = [u for u in urls if isinstance(u, str) and u.strip()]
    if not urls:
        raise RuntimeError("No ASL clips available to merge.")

    urls = urls[:20]  # cap
    local_paths = _download_clips(urls)
    if not local_paths:
        raise RuntimeError("All ASL clip downloads failed.")

    try:
        _concat_with_filter(local_paths, output_path)
    finally:
        for p in local_paths:
            try:
                os.remove(p)
            except Exception:
                pass


# --------------------------- Worker entry ----------------------------
def process_audio_worker(
    job_id: str,
    audio_path: str,
    video_jobs: dict,
    translate_text_to_sign,
    static_dir: str
):
    """
    Background job runner. Expects translate_text_to_sign(text) -> list[str] of URLs.
    Writes /videos/output_<job_id>.mp4 and updates video_jobs[job_id].
    """
    try:
        print(f"[{job_id}] transcribing…")
        transcript = transcribe_with_assemblyai(audio_path)
        print(f"[{job_id}] transcript: {transcript!r}")

        urls = translate_text_to_sign(transcript)
        print(f"[{job_id}] fetched {len(urls)} ASL clip URLs")

        out_path = os.path.join(static_dir, f"output_{job_id}.mp4")

        if urls:
            try:
                generate_merged_video(urls, out_path)
            except Exception as merge_err:
                # If concatenation fails (site blocked, mixed codecs, etc.), fall back to placeholder
                print(f"[{job_id}] merge failed, generating placeholder: {merge_err}")
                _placeholder_video(out_path, transcript)
        else:
            # No URLs found at all → placeholder so UI still completes
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
        # Clean up uploaded audio
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass
