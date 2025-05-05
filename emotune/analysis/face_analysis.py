import os
import time
from datetime import datetime
import torch
import cv2
from PIL import Image
from facenet_pytorch import MTCNN
import numpy as np
import sys
from pathlib import Path

# Add emonet to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'external', 'emonet')))
from emonet.models import EmoNet

class FaceEmotion:
    def __init__(self, model_path=None, device='cpu'):
        self.device = torch.device(device)

        if model_path is None:
            model_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__), '..', '..', 'external', 'emonet', 'pretrained', 'emonet_8.pth'))

        self.model = EmoNet(n_expression=8).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        self.mtcnn = MTCNN(keep_all=False, device=self.device)

        self.emotion_classes = {
            0: "Neutral", 1: "Happy", 2: "Sad", 3: "Surprise",
            4: "Fear", 5: "Disgust", 6: "Anger", 7: "Contempt"
        }

        os.makedirs("face_captures", exist_ok=True)

    def capture_and_save(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return None
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("Error: Failed to capture image.")
            return None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"face_captures/face_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        return filename

   
    def analyze_face(self, image_path):
        img = Image.open(image_path).convert('RGB')
        face, prob = self.mtcnn(img, return_prob=True)

        if face is None:
            print("No face detected.")
            return None

        # Convert to numpy image in [0, 255] range for OpenCV
        face_np = face.permute(1, 2, 0).cpu().numpy()  # Shape: (H, W, C), float32 in [0, 1]
        face_np = (face_np * 255).astype(np.uint8)      # Convert to uint8
        face_np = cv2.resize(face_np, (256, 256))       # Resize with correct data type

        # Convert to tensor in [0, 1] range
        face_tensor = torch.tensor(face_np).permute(2, 0, 1).float().to(self.device) / 255.0
        face_tensor = face_tensor.unsqueeze(0)

        with torch.no_grad():
            out = self.model(face_tensor)
            valence = out['valence'].clamp(-1, 1).cpu().item()
            arousal = out['arousal'].clamp(-1, 1).cpu().item()
            expression_logits = out['expression']
            expression_idx = torch.argmax(torch.nn.functional.softmax(expression_logits, dim=1)).cpu().item()
            expression_label = self.emotion_classes[expression_idx]

        return {
            'valence': valence,
            'arousal': arousal,
            'expression': expression_label
        }

    def run(self, iterations=3, delay=2):
        for i in range(iterations):
            image_path = self.capture_and_save()
            if image_path:
                result = self.analyze_face(image_path)
                if result:
                    print(f"[{i + 1}] VA: (valence={result['valence']:.3f}, arousal={result['arousal']:.3f}) → Expression: {result['expression']}")
            time.sleep(delay)

if __name__ == '__main__':
    fe = FaceEmotion(device='cuda' if torch.cuda.is_available() else 'cpu')
    fe.run(iterations=2, delay=2)
