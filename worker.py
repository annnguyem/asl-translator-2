import traceback
import os
import requests
import time

ASSEMBLYAI_API_KEY = "your_assemblyai_key_here"

def transcribe_with_assemblyai(audio_path):
    headers = {'authorization': ASSEMBLYAI_API_KEY}

    with open(audio_path, 'rb') as f:
        response = requests.post(
            'https://api.assemblyai.com/v2/upload',
            headers=headers,
            files={'file': f}
        )
    response.raise_for_status()
    upload_url = response.json()['upload_url']

    transcript_response = requests.post(
        'https://api.assemblyai.com/v2/transcript',
        json={'audio_url': upload_url},
        headers=headers
    )
    transcript_response.raise_for_status()
    transcript_id = transcript_response.json()['id']

    while True:
        polling = requests.get(f'https://api.assemblyai.com/v2/transcript/{transcript_id}', headers=headers)
        polling.raise_for_status()
        status = polling.json()['status']
        if status == 'completed':
            return polling.json()['text']
        elif status == 'error':
            raise Exception(f"AssemblyAI error: {polling.json()['error']}")
        time.sleep(3)

def process_audio_worker(job_id, audio_path, video_jobs, translate_text_to_sign):
    try:
        print(f"🎬 [{job_id}] Transcribing with AssemblyAI...")
        transcript = transcribe_with_assemblyai(audio_path)
        print(f"🗣️ Transcript: {transcript}")

        video_urls = translate_text_to_sign(transcript)
        print(f"🔗 Found {len(video_urls)} ASL video URLs.")

        video_jobs[job_id] = {
            "status": "ready",
            "video_urls": video_urls,
            "transcript": transcript
        }

    except Exception:
        traceback.print_exc()
        video_jobs[job_id]["status"] = "error"
