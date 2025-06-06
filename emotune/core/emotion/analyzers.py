import os
import sys
import time
import torch
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from datetime import datetime
import logging

# Add emonet to the path for face model
emonet_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'vendor', 'emonet'))
sys.path.append(emonet_path)
from emonet.models import EmoNet

# Helper for entropy
from scipy.stats import entropy

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceAnalyzer:
    """Simplified face emotion analyzer using EmoNet, matching legacy output"""
    def __init__(self, model_path=None, device='cpu'):
        self.device = torch.device(device)
        self.mtcnn = MTCNN(keep_all=False, device=self.device)
        if model_path is None:
            model_path = os.path.join(emonet_path, 'pretrained', 'emonet_8.pth')
        self.model = EmoNet(n_expression=8).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        state = {k.replace('module.', ''): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self.labels = ['neutral','happy','sad','surprise','fear','disgust','anger','contempt']

    def analyze_frame(self, frame: np.ndarray) -> tuple:
        # Convert OpenCV BGR frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        # Save debug frame (as RGB)
        debug_dir = 'face_debug'
        os.makedirs(debug_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        frame_path = os.path.join(debug_dir, f'frame_{ts}.jpg')
        Image.fromarray(frame_rgb).save(frame_path)
        # Face detection
        face, prob = self.mtcnn(img, return_prob=True)
        if face is None or prob < 0.8:
            boxes, _ = self.mtcnn.detect(img)
            if boxes is not None and len(boxes) > 0:
                arr_img = np.array(img)  # RGB array
                areas = [(x2-x1)*(y2-y1) for (x1, y1, x2, y2) in boxes]
                idx = int(np.argmax(areas))
                x1, y1, x2, y2 = [int(v) for v in boxes[idx]]
                crop = arr_img[y1:y2, x1:x2]
                if crop.size > 0:
                    crop = cv2.resize(crop, (256, 256))
                    crop = np.clip(crop, 0, 255).astype(np.uint8)
                    crop_path = os.path.join(debug_dir, f'crop_{ts}.jpg')
                    Image.fromarray(crop).save(crop_path)
                    tensor = torch.from_numpy(crop).float().permute(2,0,1).unsqueeze(0).to(self.device)/255.0
                    return self._process_emonet(tensor)
            dist = {l: 1/8 for l in self.labels}
            return {'valence':0.0, 'arousal':0.0, 'discrete_emotions':dist}, 0.0
        arr = (face.permute(1,2,0).cpu().numpy() * 255)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        arr = cv2.resize(arr, (256, 256))
        crop_path = os.path.join(debug_dir, f'crop_{ts}.jpg')
        Image.fromarray(arr).save(crop_path)
        tensor = torch.from_numpy(arr).float().permute(2,0,1).unsqueeze(0).to(self.device)/255.0
        return self._process_emonet(tensor)

    def _process_emonet(self, tensor: torch.Tensor) -> tuple:
        """Shared EmoNet processing for both paths"""
        with torch.no_grad():
            out = self.model(tensor)
            probs = torch.softmax(out['expression'], dim=1).cpu().numpy()[0]
            dist = dict(zip(self.labels, probs.tolist()))
            val = float(out['valence'].clamp(-1,1).item())
            aro = float(out['arousal'].clamp(-1,1).item())
            conf = max(0.0, min(1.0, 1 - entropy(probs) / np.log(len(probs))))
        return {'valence':val, 'arousal':aro, 'discrete_emotions':dist}, conf

class VoiceAnalyzer:
    """Simplified voice emotion analyzer using Wav2Vec2, matching legacy output"""
    def __init__(self, model_name='audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim', device='cpu'):
        self.device = torch.device(device)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def analyze_audio(self, audio: np.ndarray, sample_rate:int=16000) -> tuple:
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        inputs = self.processor(audio, sampling_rate=sample_rate, return_tensors='pt', padding=True)
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
            preds = torch.tanh(logits).cpu().numpy()[0]
        # Audeering: [arousal, dominance, valence]
        val, aro = float(preds[2]), float(preds[0])
        # confidence as mean abs
        conf = float(np.mean(np.abs([val, aro])))
        conf = max(0.3, min(1.0, conf))
        return {'valence':val, 'arousal':aro}, conf

class EmotionAnalyzer:
    """Combined analyzer for face and voice matching legacy structure"""
    def __init__(self, device='cpu'):
        self.face = FaceAnalyzer(device=device)
        self.voice = VoiceAnalyzer(device=device)

    def analyze(self, frame: np.ndarray=None, audio: np.ndarray=None, sr:int=16000) -> dict:
        res = {'timestamp': time.time(), 'face': None, 'voice': None}
        if frame is not None:
            em, cf = self.face.analyze_frame(frame)
            res['face'] = {'emotions': em, 'confidence': cf}
        if audio is not None:
            em, cf = self.voice.analyze_audio(audio, sample_rate=sr)
            res['voice'] = {'emotions': em, 'confidence': cf}
        return res

