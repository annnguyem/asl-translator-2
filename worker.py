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

    # Request transcript with word-level timestamps
    transcript_request = {
        'audio_url': upload_url,
        'punctuate': True,
        'format_text': True,
        'word_boost': [],  # optional
        'word_timestamps': True  # <-- Enable word timestamps
    }

    transcript_response = requests.post(
        'https://api.assemblyai.com/v2/transcript',
        json=transcript_request,
        headers=headers
    )
    transcript_response.raise_for_status()
    transcript_id = transcript_response.json()['id']

    while True:
        polling = requests.get(f'https://api.assemblyai.com/v2/transcript/{transcript_id}', headers=headers)
        polling.raise_for_status()
        data = polling.json()
        status = data['status']
        if status == 'completed':
            return data  # return full transcript JSON with words
        elif status == 'error':
            raise Exception(f"AssemblyAI error: {data['error']}")
        time.sleep(3)


def process_audio_worker(job_id, audio_path, video_jobs, translate_text_to_sign, generate_merged_video, static_dir):
    try:
        print(f"🎬 [{job_id}] Transcribing with AssemblyAI...")
        transcript_data = transcribe_with_assemblyai(audio_path)

        transcript = transcript_data.get('text', '')
        words = transcript_data.get('words', [])
        print(f"🗣️ Transcript: {transcript}")
        print(f"🕒 Words with timestamps: {words}")

        # Translate transcript text into ASL video paths or URLs
        video_urls = translate_text_to_sign(transcript)
        print(f"🔗 Found {len(video_urls)} ASL video URLs.")

        # Generate the merged video with timing
        output_path = os.path.join(static_dir, f"output_{job_id}.mp4")
        generate_merged_video(video_urls, words, output_path)

        video_jobs[job_id] = {
            "status": "ready",
            "video_urls": video_urls,
            "transcript": transcript
        }

    except Exception:
        traceback.print_exc()
        video_jobs[job_id]["status"] = "error"
