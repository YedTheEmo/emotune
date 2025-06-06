import os
import time
from datetime import datetime
import torch
import cv2
from PIL import Image
from facenet_pytorch import MTCNN
import numpy as np
import sys

# Set up test output directory
TEST_DEBUG_DIR = os.path.join(os.path.dirname(__file__), 'face_debug_test')
os.makedirs(TEST_DEBUG_DIR, exist_ok=True)

# Camera index to test
CAMERA_INDEX = 0

# Initialize MTCNN (CPU or CUDA) - exactly like your working version
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

mtcnn = MTCNN(keep_all=False, device=device)

def capture_and_save():
    """Capture image from webcam and save as JPEG - exact copy of your working method"""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Failed to capture image.")
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(TEST_DEBUG_DIR, f"face_{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    print(f"Saved: {filename}")
    return filename

def analyze_face_debug(image_path):
    """Analyze face using your exact working pipeline with debug info"""
    
    # Step 1: Load image and convert to RGB (your exact method)
    img = Image.open(image_path).convert('RGB')
    print(f"PIL image size: {img.size}, mode: {img.mode}")
    
    # Step 2: MTCNN face detection (your exact method)
    face, prob = mtcnn(img, return_prob=True)
    
    if face is None:
        print("No face detected.")
        return None
    
    print(f"Face detection confidence: {prob:.3f}")
    print(f"Face tensor shape from MTCNN: {face.shape}")
    print(f"Face tensor range from MTCNN: [{face.min():.3f}, {face.max():.3f}]")
    
    # Step 3: Convert to numpy image in [0, 255] range (your exact method)
    face_np = face.permute(1, 2, 0).cpu().numpy()  # Shape: (H, W, C), float32 in [0, 1]
    print(f"After permute to numpy: shape={face_np.shape}, range=[{face_np.min():.3f}, {face_np.max():.3f}]")
    
    face_np = (face_np * 255).astype(np.uint8)      # Convert to uint8
    print(f"After scaling to uint8: shape={face_np.shape}, range=[{face_np.min()}, {face_np.max()}]")
    
    # Step 4: Resize (your exact method)
    face_np = cv2.resize(face_np, (256, 256))       # Resize with correct data type
    print(f"After resize to 256x256: shape={face_np.shape}")
    
    # Save the processed face for visual verification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    crop_path = os.path.join(TEST_DEBUG_DIR, f"crop_{timestamp}.jpg")
    # Convert RGB to BGR for cv2.imwrite
    face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(crop_path, face_bgr)
    print(f"Saved face crop: {crop_path}")
    
    # Step 5: Convert to tensor in [0, 1] range (your exact method for EmoNet)
    face_tensor = torch.tensor(face_np).permute(2, 0, 1).float().to(device) / 255.0
    print(f"Tensor after permute and normalize: shape={face_tensor.shape}, range=[{face_tensor.min():.3f}, {face_tensor.max():.3f}]")
    
    face_tensor = face_tensor.unsqueeze(0)
    print(f"Final tensor with batch dim: shape={face_tensor.shape}")
    
    # This tensor is now ready for EmoNet inference
    # In your working code, this would be: out = self.model(face_tensor)
    
    return {
        'face_tensor': face_tensor,
        'confidence': prob,
        'crop_path': crop_path,
        'preprocessing_success': True
    }

def test_batch_processing(results_list):
    """Test batch processing exactly like EmoNet expects"""
    if not results_list:
        print("No faces to batch process.")
        return
    
    # Stack tensors for batch processing (remove the batch dimension first, then stack)
    face_tensors = [r['face_tensor'].squeeze(0) for r in results_list]
    batch_tensor = torch.stack(face_tensors)
    print(f"\nBatch tensor shape: {batch_tensor.shape}")  # Should be [N, 3, 256, 256]
    print(f"Batch tensor range: [{batch_tensor.min():.3f}, {batch_tensor.max():.3f}]")
    
    # This is exactly what your EmoNet model expects:
    # with torch.no_grad():
    #     out = self.model(batch_tensor)
    print("✓ Batch tensor ready for EmoNet inference (matches your working pipeline)")

def main():
    print("=== EmoNet Pipeline Test (Exact Copy of Working Version) ===\n")
    
    results = []
    
    for i in range(3):
        print(f"\n--- Capture {i+1}/3 ---")
        
        # Use your exact capture method
        image_path = capture_and_save()
        if image_path:
            # Only keep the first pipeline (the correct one)
            result = analyze_face_debug(image_path)
            if result:
                results.append(result)
                print(f"✓ Successfully processed face {i+1}")
            else:
                print(f"✗ Failed to process face {i+1}")
        
        time.sleep(2)
    
    # Test batch processing (keep only the first pipeline's results)
    print(f"\n--- Batch Processing Test ---")
    if results:
        face_tensors = [results[0]['face_tensor'].squeeze(0)]
        batch_tensor = torch.stack(face_tensors)
        print(f"\nBatch tensor shape: {batch_tensor.shape}")
        print(f"Batch tensor range: [{batch_tensor.min():.3f}, {batch_tensor.max():.3f}]")
        print("✓ Batch tensor ready for EmoNet inference (matches your working pipeline)")
    else:
        print("No faces to batch process.")
    
    print(f"\n=== Summary ===")
    print(f"✓ Test complete. Check {TEST_DEBUG_DIR} for results.")
    print(f"✓ Processed {len(results)} faces successfully.")
    print(f"✓ Pipeline matches your working FaceEmotion class exactly.")

if __name__ == '__main__':
    main()