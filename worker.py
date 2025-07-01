import traceback
import os
import subprocess
import requests
import time

ASSEMBLYAI_API_KEY = "2b791d89824a4d5d8eeb7e310aa6542f"

def transcribe_with_assemblyai(audio_path):
    headers = {'authorization': ASSEMBLYAI_API_KEY}

    # 1. Upload the audio file
    with open(audio_path, 'rb') as f:
        response = requests.post(
            'https://api.assemblyai.com/v2/upload',
            headers=headers,
            files={'file': f}
        )
    response.raise_for_status()
    upload_url = response.json()['upload_url']

    # 2. Submit transcription request
    transcript_response = requests.post(
        'https://api.assemblyai.com/v2/transcript',
        json={'audio_url': upload_url},
        headers=headers
    )
    transcript_response.raise_for_status()
    transcript_id = transcript_response.json()['id']

    # 3. Poll for result
    while True:
        polling = requests.get(f'https://api.assemblyai.com/v2/transcript/{transcript_id}', headers=headers)
        polling.raise_for_status()
        status = polling.json()['status']
        if status == 'completed':
            return polling.json()['text']
        elif status == 'error':
            raise Exception(f"AssemblyAI error: {polling.json()['error']}")
        time.sleep(3)

def process_audio_worker(job_id, temp_audio_path, video_jobs, STATIC_DIR, translate_text_to_sign, generate_merged_video):
    try:
        model = WhisperModel("tiny", compute_type="int8")

        print(f"🎬 [{job_id}] Starting transcription...")

        sentence = " ".join([seg.text for seg in segments])
        print(f"🗣️ Transcript: {sentence}")

        paths = translate_text_to_sign(sentence)
        print(f"🎞️ ASL clips found: {len(paths)}")
        print(f"🧾 Paths: {paths}")

        if not paths:
            video_jobs[job_id]["status"] = "error"
            return

        # Make sure all clips exist
        valid_paths = []
        for path in paths:
            print(f"📂 Checking: {path}")
            if os.path.isfile(path):
                valid_paths.append(path)
            else:
                print(f"❌ Missing clip: {path}")

        if not valid_paths:
            print("⚠️ No valid clips to merge.")
            video_jobs[job_id]["status"] = "error"
            return

        output_path = os.path.join(STATIC_DIR, f"output_{job_id}.mp4")
        generate_merged_video(valid_paths, output_path)

        # Mark job as ready
        
        open(output_path.replace(".mp4", ".done"), "w").close()
        video_jobs[job_id] = {
            "status": "ready",
            "path": output_path,
            "transcript": sentence
        }
        print(f"✅ [{job_id}] Video ready at {output_path}")
        print(f"✅ [{job_id}] Transcript: {sentence}")

    except Exception:
        traceback.print_exc()
        video_jobs[job_id]["status"] = "error"
