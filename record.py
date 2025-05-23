import sounddevice as sd
import numpy as np
import wave
import subprocess
import os
import time
from datetime import datetime

# Define constants
SAMPLE_RATE = 16000
DURATION = 10  # Record for 10 seconds


def record_audio(duration, sample_rate):
    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")
    return audio_data

def save_wav(audio_data, filename, sample_rate):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

def convert_wav_to_flac(wav_file, flac_file):
    subprocess.run(["ffmpeg", "-y", "-i", wav_file, "-ac", "1", "-ar", str(SAMPLE_RATE), flac_file],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == "__main__":
    name = input("What is your name? ").strip()

    # Create main directory for the user
    main_dir = os.path.join(os.getcwd(), name)
    os.makedirs(main_dir, exist_ok=True)

    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sub_dir = os.path.join(main_dir, timestamp)
    os.makedirs(sub_dir, exist_ok=True)

    sequence = 1  # Start sequence number

    while True:  # Loop for continuous recording (exit manually if needed)
        # Generate filenames
        wav_filename = os.path.join(sub_dir, f"{name}-{timestamp}-{sequence}.wav")
        flac_filename = os.path.join(sub_dir, f"{name}-{timestamp}-{sequence}.flac")

        # Record and save audio
        audio_data = record_audio(DURATION, SAMPLE_RATE)
        save_wav(audio_data, wav_filename, SAMPLE_RATE)

        # Convert WAV to FLAC
        convert_wav_to_flac(wav_filename, flac_filename)

        print(f"Saved: {flac_filename}")

        sequence += 1
        time.sleep(1)  # Short delay before next recording
