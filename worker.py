import traceback
import os
import requests
import time

ASSEMBLYAI_API_KEY = "your_assemblyai_key_here"

def transcribe_with_assemblyai(audio_path):
    headers = {'authorization': ASSEMBLYAI_API_KEY}

    print(f"⏳ Uploading audio for transcription: {audio_path}", flush=True)
    with open(audio_path, 'rb') as f:
        response = requests.post(
            'https://api.assemblyai.com/v2/upload',
            headers=headers,
            files={'file': f}
        )
    response.raise_for_status()
    upload_url = response.json()['upload_url']
    print(f"✅ Uploaded audio. Upload URL: {upload_url}", flush=True)

    # Request transcript with word-level timestamps
    transcript_request = {
        'audio_url': upload_url,
        'punctuate': True,
        'format_text': True,
        'word_boost': [],  # optional
        'word_timestamps': True  # <-- Enable word timestamps
    }

    print("⏳ Requesting transcript with word timestamps...", flush=True)
    transcript_response = requests.post(
        'https://api.assemblyai.com/v2/transcript',
        json=transcript_request,
        headers=headers
    )
    transcript_response.raise_for_status()
    transcript_id = transcript_response.json()['id']
    print(f"✅ Transcript requested. Transcript ID: {transcript_id}", flush=True)

    # Polling for transcript completion
    while True:
        polling = requests.get(f'https://api.assemblyai.com/v2/transcript/{transcript_id}', headers=headers)
        polling.raise_for_status()
        data = polling.json()
        status = data['status']
        print(f"🔄 Polling transcription status: {status}", flush=True)
        if status == 'completed':
            print("✅ Transcription completed.", flush=True)
            return data  # return full transcript JSON with words
        elif status == 'error':
            raise Exception(f"AssemblyAI error: {data['error']}")
        time.sleep(3)


def process_audio_worker(job_id, audio_path, video_jobs, translate_text_to_sign, generate_merged_video, static_dir):
    try:
        print(f"🎬 [{job_id}] Starting transcription and video generation workflow...", flush=True)
        transcript_data = transcribe_with_assemblyai(audio_path)

        transcript = transcript_data.get('text', '')
        words = transcript_data.get('words', [])
        print(f"🗣️ Transcript text: {transcript}", flush=True)
        print(f"🕒 Word timestamps received: {len(words)} words", flush=True)

        # Log first few words with timestamps for sanity check
        for w in words[:5]:
            start_ms = w.get('start', 'N/A')
            end_ms = w.get('end', 'N/A')
            word_text = w.get('text', '')
            print(f"   Word: '{word_text}' start: {start_ms}ms end: {end_ms}ms", flush=True)

        # Translate transcript text into ASL video URLs
        video_urls = translate_text_to_sign(transcript)
        print(f"🔗 Retrieved {len(video_urls)} ASL video URLs for translation.", flush=True)

        # Generate merged video with timing adjustment
        output_path = os.path.join(static_dir, f"output_{job_id}.mp4")
        print(f"🎥 Generating merged video at: {output_path}", flush=True)
        generate_merged_video(video_urls, words, output_path)

        video_jobs[job_id] = {
            "status": "ready",
            "video_urls": video_urls,
            "transcript": transcript
        }
        print(f"✅ Job [{job_id}] completed successfully.", flush=True)

    except Exception:
        print(f"❌ Error occurred during processing job [{job_id}]:", flush=True)
        traceback.print_exc()
