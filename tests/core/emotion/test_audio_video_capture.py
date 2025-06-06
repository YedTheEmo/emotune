import cv2
import sounddevice as sd
import numpy as np
from PIL import Image
import time
import os
from datetime import datetime

# Output directory for debug images
DEBUG_DIR = os.path.join(os.path.dirname(__file__), 'test_audio_video_capture_debug')
os.makedirs(DEBUG_DIR, exist_ok=True)

CAMERA_INDEX = 0
SAMPLE_RATE = 16000
AUDIO_DURATION = 1.0  # seconds


def capture_frame():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None
    # Warmup
    for i in range(5):
        ret, warmup_frame = cap.read()
        print(f"Warmup {i+1}/5: ret={ret}, mean={np.mean(warmup_frame) if ret and warmup_frame is not None else 'N/A'}")
        time.sleep(0.1)
    time.sleep(1.0)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print("Error: Failed to capture frame.")
        return None
    mean_val = np.mean(frame)
    print(f"Captured frame: shape={frame.shape}, mean={mean_val}")
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    frame_path = os.path.join(DEBUG_DIR, f'frame_{ts}.jpg')
    cv2.imwrite(frame_path, frame)
    print(f"Saved frame: {frame_path}")
    return frame, mean_val, frame_path

def capture_audio():
    print("Capturing audio...")
    audio = sd.rec(int(AUDIO_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    print(f"Captured audio: shape={audio.shape}, mean={np.mean(audio)}")
    return audio

def main():
    print("=== Minimal Audio+Video Capture Test ===\n")
    # 1. Capture frame only
    print("[1] Video only:")
    frame, mean_val, _ = capture_frame()
    print(f"Frame mean (video only): {mean_val}")
    time.sleep(1)
    # 2. Capture audio only
    print("\n[2] Audio only:")
    audio = capture_audio()
    time.sleep(1)
    # 3. Capture frame while audio stream is open
    print("\n[3] Video capture while audio stream is open:")
    def audio_callback(indata, frames, time_info, status):
        pass  # No-op
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=audio_callback):
        print("Audio stream started.")
        frame, mean_val, _ = capture_frame()
        print(f"Frame mean (with audio stream): {mean_val}")
        time.sleep(1)
    print("Audio stream closed.")
    print("\nDone. Check the debug directory for saved frames.")

if __name__ == '__main__':
    main()
