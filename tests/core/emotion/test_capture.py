import os
import time
from datetime import datetime
import cv2
import numpy as np
from PIL import Image

# Set up debug output directory
DEBUG_DIR = os.path.join(os.path.dirname(__file__), 'face_debug_capturepy')
os.makedirs(DEBUG_DIR, exist_ok=True)

CAMERA_INDEX = 0


def capture_and_save_like_capturepy():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    # Mimic capture.py: warm up and discard 5 frames
    for i in range(5):
        ret, warmup_frame = cap.read()
        print(f"Warmup read {i+1}/5: ret={ret}, mean={np.mean(warmup_frame) if ret and warmup_frame is not None else 'N/A'}")
        time.sleep(0.1)
    time.sleep(1.0)

    # Try to read a frame (like in capture.py main loop)
    ret, frame = cap.read()
    print(f"Main capture: ret={ret}, shape={getattr(frame, 'shape', None)}, mean={np.mean(frame) if ret and frame is not None else 'N/A'}")
    cap.release()

    if not ret or frame is None:
        print("Error: Failed to capture image.")
        return None
    if not isinstance(frame, np.ndarray):
        print(f"Error: Frame is not ndarray, got {type(frame)}")
        return None
    if frame.ndim != 3 or frame.shape[2] != 3:
        print(f"Error: Unexpected frame shape: {frame.shape}")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    frame_path = os.path.join(DEBUG_DIR, f"frame_{timestamp}.jpg")
    cv2.imwrite(frame_path, frame)
    print(f"Saved BGR frame: {frame_path}")

    # Convert to RGB and PIL Image (like analyzers.py)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    pil_path = os.path.join(DEBUG_DIR, f"frame_{timestamp}_pil.jpg")
    img.save(pil_path)
    print(f"Saved PIL RGB frame: {pil_path}")

    return frame, frame_path, pil_path


def main():
    print("=== Test: capture.py Camera Logic Extraction ===\n")
    for i in range(3):
        print(f"\n--- Capture {i+1}/3 ---")
        capture_and_save_like_capturepy()
        time.sleep(2)
    print(f"\nCheck {DEBUG_DIR} for saved frames.")

if __name__ == '__main__':
    main()
