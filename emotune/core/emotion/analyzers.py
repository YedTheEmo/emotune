import os
import sys
import time
import torch
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

# Add emonet to the path for face model
emonet_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'vendor', 'emonet'))
sys.path.append(emonet_path)
from emonet.models import EmoNet

# Helper for entropy
from scipy.stats import entropy

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
        # Convert and detect
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        face, prob = self.mtcnn(img, return_prob=True)
        if face is None or prob < 0.8:
            # default emotions
            dist = {l: 1/8 for l in self.labels}
            return {'valence':0.0, 'arousal':0.0, 'discrete_emotions':dist}, 0.0
        # preprocess
        arr = (face.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        arr = cv2.resize(arr, (256,256))
        tensor = torch.from_numpy(arr).float().permute(2,0,1).unsqueeze(0).to(self.device)/255.0
        # model forward
        with torch.no_grad():
            out = self.model(tensor)
            # discrete probabilities
            probs = torch.softmax(out['expression'], dim=1).cpu().numpy()[0]
            dist = dict(zip(self.labels, probs.tolist()))
            # valence/arousal
            val = float(out['valence'].clamp(-1,1).item())
            aro = float(out['arousal'].clamp(-1,1).item())
            # confidence via entropy
            conf = float(1.0 - entropy(probs) / np.log(len(probs)))
            conf = max(0.0, min(1.0, conf))
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

