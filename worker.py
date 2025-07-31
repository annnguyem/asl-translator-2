import os
import time
import logging
import traceback
import requests

# 🔧 Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

ASSEMBLYAI_API_KEY = "2b791d89824a4d5d8eeb7e310aa6542f"

def transcribe_with_assemblyai(audio_path):
    headers = {    "authorization": ASSEMBLYAI_API_KEY,
    "transfer-encoding": "chunked"}

    logging.info(f"⏳ Uploading audio for transcription: {audio_path}")
    if not os.path.exists(audio_path):
        logging.error(f"❌ Audio path does not exist: {audio_path}")
    elif os.path.getsize(audio_path) < 1000:
        logging.error(f"❌ Audio file too small ({os.path.getsize(audio_path)} bytes): {audio_path}")
    else:
        logging.info(f"✅ Ready to upload {audio_path} ({os.path.getsize(audio_path)} bytes)")
    with open(audio_path, 'rb') as f:
        response = requests.post(
            'https://api.assemblyai.com/v2/upload',
            headers=headers,
            data=f
        )
    response.raise_for_status()
    upload_url = response.json()['upload_url']
    logging.info(f"✅ Uploaded audio. Upload URL: {upload_url}")

    # Request transcript with word-level timestamps
    transcript_request = {
        'audio_url': upload_url,
        'punctuate': True,
        'format_text': True,
        'word_boost': [],
        'word_timestamps': True
    }

    logging.info("⏳ Requesting transcript with word timestamps...")
    transcript_response = requests.post(
        'https://api.assemblyai.com/v2/transcript',
        json=transcript_request,
        headers=headers
    )
    transcript_response.raise_for_status()
    transcript_id = transcript_response.json()['id']
    logging.info(f"✅ Transcript requested. Transcript ID: {transcript_id}")

    # Poll for completion
    while True:
        polling = requests.get(f'https://api.assemblyai.com/v2/transcript/{transcript_id}', headers=headers)
        polling.raise_for_status()
        data = polling.json()
        status = data['status']
        logging.info(f"🔄 Polling transcription status: {status}")
        if status == 'completed':
            logging.info("✅ Transcription completed.")
            return data
        elif status == 'error':
            raise Exception(f"AssemblyAI error: {data['error']}")
        time.sleep(3)


def process_audio_worker(job_id, audio_path, video_jobs, translate_text_to_sign, generate_merged_video, static_dir):
    try:
        logging.info(f"🎬 [{job_id}] Starting transcription and video generation workflow...")
        file_size = os.path.getsize(audio_path)
        logging.info(f"📁 Uploading audio file size: {file_size} bytes")
        
        if file_size == 0:
            raise ValueError("❌ Audio file is empty. Cannot upload to AssemblyAI.")

        transcript_data = transcribe_with_assemblyai(audio_path)

        transcript = transcript_data.get('text', '')
        words = transcript_data.get('words', [])
        logging.info(f"🗣️ Transcript text: {transcript}")
        logging.info(f"🕒 Word timestamps received: {len(words)} words")

        for w in words[:5]:
            start_ms = w.get('start', 'N/A')
            end_ms = w.get('end', 'N/A')
            word_text = w.get('text', '')
            logging.info(f"   Word: '{word_text}' start: {start_ms}ms end: {end_ms}ms")

        video_urls = translate_text_to_sign(transcript)
        logging.info(f"🔗 Retrieved {len(video_urls)} ASL video URLs for translation.")

        output_path = os.path.join(static_dir, f"output_{job_id}.mp4")
        logging.info(f"🎥 Generating merged video at: {output_path}")
        word_timestamps = []
        for w in words:
            word_clean = w.get("text", "").strip().lower()
            word_timestamps.append({
                "word": word_clean,
                "start": w.get("start", 0),
                "end": w.get("end", 0)
            })
        generate_merged_video([v["url"] for v in video_url_map], word_timestamps, output_path)

        video_jobs[job_id] = {
            "status": "ready",
            "video_urls": video_urls,
            "transcript": transcript
        }
        logging.info(f"✅ Job [{job_id}] completed successfully.")

    except Exception:
        logging.error(f"❌ Error occurred during processing job [{job_id}]:")
        logging.exception("Exception traceback:")
