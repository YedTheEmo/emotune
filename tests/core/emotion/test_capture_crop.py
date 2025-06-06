import os
import time
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import torch

# Set up debug output directory
DEBUG_DIR = os.path.join(os.path.dirname(__file__), 'face_debug_capturepy_full')
os.makedirs(DEBUG_DIR, exist_ok=True)

CAMERA_INDEX = 0

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
mtcnn = MTCNN(keep_all=False, device=device)

def capture_and_crop_like_pipeline():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    # Warm up: discard 5 frames
    for i in range(5):
        ret, warmup_frame = cap.read()
        print(f"Warmup read {i+1}/5: ret={ret}, mean={np.mean(warmup_frame) if ret and warmup_frame is not None else 'N/A'}")
        time.sleep(0.1)
    time.sleep(1.0)

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

    # --- Face detection and cropping (as in analyzers.py) ---
    face, prob = mtcnn(img, return_prob=True)
    print(f"MTCNN face detection confidence: {prob}")
    crop_path = None
    if face is None or (prob is not None and prob < 0.8):
        print("Low face detection confidence or no face detected, attempting fallback.")
        boxes, _ = mtcnn.detect(img)
        if boxes is not None and len(boxes) > 0:
            arr_img = np.array(img)
            areas = [(x2-x1)*(y2-y1) for (x1, y1, x2, y2) in boxes]
            idx = int(np.argmax(areas))
            x1, y1, x2, y2 = [int(v) for v in boxes[idx]]
            crop = arr_img[y1:y2, x1:x2]
            if crop.size > 0:
                crop = cv2.resize(crop, (256,256))
                crop_path = os.path.join(DEBUG_DIR, f"crop_{timestamp}.jpg")
                cv2.imwrite(crop_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                print(f"Saved fallback crop: {crop_path}")
            else:
                print("Fallback crop is empty.")
        else:
            print("No face detected in fallback.")
    else:
        arr = (face.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        arr = cv2.resize(arr, (256,256))
        crop_path = os.path.join(DEBUG_DIR, f"crop_{timestamp}.jpg")
        cv2.imwrite(crop_path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        print(f"Saved main path crop: {crop_path}")

    return frame_path, pil_path, crop_path

def main():
    print("=== Test: capture.py + analyzers.py Camera & Cropping Logic Extraction ===\n")
    for i in range(3):
        print(f"\n--- Capture {i+1}/3 ---")
        capture_and_crop_like_pipeline()
        time.sleep(2)
    print(f"\nCheck {DEBUG_DIR} for saved frames and crops.")

if __name__ == '__main__':
    main()
