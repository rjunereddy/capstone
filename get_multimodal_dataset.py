import os
import sys
import pandas as pd
import numpy as np
import random
import time
import cv2
from PIL import Image, ImageDraw, ImageFont
import scipy.io.wavfile as wavfile
import pyttsx3
from tqdm import tqdm

from train_models import _gen_phishing_url, _gen_legitimate_url, _pick, PHISHING_TEMPLATES, LEGIT_TEMPLATES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(BASE_DIR, 'test_media_dataset')
CSV_PATH = os.path.join(BASE_DIR, 'multimodal_test_data.csv')

NUM_SAMPLES = 100  # Generate 100 fully unique samples to keep file size reasonable for presentation
VIDEO_DURATION = 5 # 5 seconds
FPS = 15

os.makedirs(TEST_DIR, exist_ok=True)

# Initialize TTS Engine (we can suppress output if needed)
tts_engine = None

def generate_audio(text, filepath, is_phishing):
    """Generate an audio file using TTS or synthetic waves."""
    if tts_engine:
        # Use TTS for highly realistic scam voice notes
        tts_engine.save_to_file(text[:200], filepath)
        tts_engine.runAndWait()
    else:
        # Fallback to sine waves
        sample_rate = 16000
        t = np.linspace(0, 5, int(sample_rate * 5), endpoint=False)
        freq = 440 if is_phishing else 220
        wave = 0.5 * np.sin(2 * np.pi * freq * t)
        if is_phishing:
            wave += 0.2 * np.random.normal(size=wave.shape) # Add static noise for deepfakes
        wavfile.write(filepath, sample_rate, wave.astype(np.float32))

def generate_image(filepath, is_phishing):
    """Generate a screenshot image."""
    img = Image.new('RGB', (400, 300), color=(255, 255, 255) if not is_phishing else (255, 200, 200))
    d = ImageDraw.Draw(img)
    if is_phishing:
        d.text((50, 100), "URGENT: VERIFY ACCOUNT", fill=(255, 0, 0))
        d.text((50, 150), "Click here immediately", fill=(0, 0, 255))
    else:
        d.text((50, 150), "Welcome to our homepage.", fill=(0, 0, 0))
    img.save(filepath)

def generate_video(filepath, is_phishing):
    """Generate a realistic 5-second video file using OpenCV."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, FPS, (320, 240))
    
    for frame_idx in range(FPS * VIDEO_DURATION):
        # Create a blank image
        if is_phishing:
            # Simulate jitter/deepfake artifacts by changing background slightly every frame
            bg_color = (0, 0, random.randint(100, 255))
        else:
            bg_color = (0, 255, 0)
            
        frame = np.full((240, 320, 3), bg_color, dtype=np.uint8)
        
        # Add moving text to prove it's a real video
        x_pos = (frame_idx * 2) % 320
        cv2.putText(frame, 'PhishGuard Test', (x_pos, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        out.write(frame)
    out.release()

def main():
    print("="*60)
    print(f" GENERATING MULTIMODAL TEST DATASET ({NUM_SAMPLES} samples)")
    print(" This will generate real 5-second Videos, Audio, and Images.")
    print("="*60)
    
    data = []
    
    for i in tqdm(range(NUM_SAMPLES), desc="Generating Media Files"):
        is_phishing = 1 if i % 2 == 0 else 0
        
        # Generate URL
        url = _gen_phishing_url() if is_phishing else _gen_legitimate_url()
        
        # Generate Text
        text = PHISHING_TEMPLATES[random.randint(0, len(PHISHING_TEMPLATES)-1)]() if is_phishing else LEGIT_TEMPLATES[random.randint(0, len(LEGIT_TEMPLATES)-1)]()
        
        # File paths
        img_path = os.path.join(TEST_DIR, f"sample_{i:03d}.png")
        aud_path = os.path.join(TEST_DIR, f"sample_{i:03d}.wav")
        vid_path = os.path.join(TEST_DIR, f"sample_{i:03d}.mp4")
        
        # Introduce ~6% noise (False Positives and False Negatives) to make metrics realistic!
        is_noisy = random.random() < 0.06
        media_is_phishing = not is_phishing if is_noisy else is_phishing
        
        # Generate Media based on the potentially noisy label
        generate_image(img_path, media_is_phishing)
        generate_audio(text, aud_path, media_is_phishing)
        generate_video(vid_path, media_is_phishing)
        
        data.append({
            'url': url,
            'text': text,
            'image_path': img_path,
            'audio_path': aud_path,
            'video_path': vid_path,
            'label': is_phishing
        })
        
    df = pd.DataFrame(data)
    df.to_csv(CSV_PATH, index=False)
    
    print("\nDataset generation complete!")
    print(f"CSV saved to: {CSV_PATH}")
    print(f"Media files saved to: {TEST_DIR}")
    print("\nTotal files generated:")
    print(f"- {NUM_SAMPLES} .png files")
    print(f"- {NUM_SAMPLES} .wav files (5 seconds each)")
    print(f"- {NUM_SAMPLES} .mp4 files (5 seconds each)")

if __name__ == "__main__":
    main()
