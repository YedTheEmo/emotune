import os
import time
import torch
import cv2
from PIL import Image
from facenet_pytorch import MTCNN
import numpy as np
import sys
from pathlib import Path
import logging
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2PreTrainedModel
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model
import torchaudio.transforms as T

# Add emonet to the path for face model
try:
    emonet_path = Path(__file__).parent.parent.parent.parent / 'vendor' / 'emonet'
    sys.path.append(str(emonet_path))
    from emonet.models import EmoNet
except ImportError as e:
    print(f"Error importing EmoNet: {e}")
    print("Please ensure the emonet submodule is available in the vendor directory.")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegressionHead(nn.Module):
    """Regression head for Wav2Vec2."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)
        return hidden_states, logits


class FaceAnalyzer:
    """Simplified face emotion analyzer using EmoNet, matching legacy output"""
    def __init__(self, model_path=None, device='cpu'):
        self.device = torch.device(device)
        self.mtcnn = MTCNN(keep_all=False, device=self.device)

        if model_path is None:
            model_path = os.path.join(emonet_path, 'pretrained', 'emonet_8.pth')
        self.model = EmoNet(n_expression=8).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.labels = ['neutral','happy','sad','surprise','fear','disgust','anger','contempt']

    def analyze_frame(self, frame: np.ndarray) -> tuple:
        logger.debug("[FaceAnalyzer] --- New Frame Analysis ---")
        try:
            # Step 1: Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            logger.debug(f"[FaceAnalyzer] Step 1: Frame converted to RGB, shape={frame_rgb.shape}")

            # Step 2: Face detection
            boxes, _ = self.mtcnn.detect(img)
            logger.debug(f"[FaceAnalyzer] Step 2: Detected {len(boxes) if boxes is not None else 0} faces.")

            if boxes is not None and len(boxes) > 0:
                # Step 3: Get the largest face
                areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
                idx = np.argmax(areas)
                x1, y1, x2, y2 = [int(b) for b in boxes[idx]]
                logger.debug(f"[FaceAnalyzer] Step 3: Largest face selected with box: ({x1}, {y1}, {x2}, {y2})")

                # Step 4: Crop face
                face_crop = frame_rgb[y1:y2, x1:x2]

                if face_crop.size > 0:
                    logger.debug(f"[FaceAnalyzer] Step 4: Cropped face size: {face_crop.shape}")
                    # Step 5: Preprocess and process with EmoNet
                    face_tensor = self._preprocess_face(face_crop)
                    em, cf = self._process_emonet(face_tensor)
                    logger.debug("[FaceAnalyzer] Analysis successful, returning result.")
                    return em, cf
                else:
                    logger.warning("[FaceAnalyzer] Step 4: Cropped face has zero size. Returning neutral.")
            else:
                logger.warning("[FaceAnalyzer] No faces detected. Returning neutral.")
            
            return {'valence': 0.0, 'arousal': 0.0}, 0.0

        except Exception as e:
            logger.error(f"[FaceAnalyzer] --- Analysis failed: {e} ---", exc_info=True)
            return {'valence': 0.0, 'arousal': 0.0}, 0.0

    def _preprocess_face(self, face_crop: np.ndarray) -> torch.Tensor:
        """Resize and normalize face image for EmoNet"""
        face_resized = cv2.resize(face_crop, (256, 256))
        face_tensor = torch.from_numpy(face_resized).permute(2, 0, 1).float().to(self.device) / 255.0
        face_tensor = face_tensor.unsqueeze(0)
        logger.debug(f"[FaceAnalyzer] Preprocessed face tensor, shape={face_tensor.shape}")
        return face_tensor

    def _process_emonet(self, tensor: torch.Tensor) -> tuple:
        """Shared EmoNet processing for both paths"""
        with torch.no_grad():
            out = self.model(tensor)
            probs = torch.softmax(out['expression'], dim=1).cpu().numpy()[0]
            dist = {self.labels[i]: f"{probs[i]:.3f}" for i in range(len(self.labels))}
            
            em = {
                'valence': out['valence'].cpu().item(),
                'arousal': out['arousal'].cpu().item(),
            }
            conf = float(np.max(probs))
            
            logger.debug(f"[FaceAnalyzer] Raw output: V={em['valence']:.3f}, A={em['arousal']:.3f}, Expression={dist}, Conf={conf:.3f}")
        return em, conf


class VoiceAnalyzer:
    """Voice emotion analyzer using Wav2Vec2, matching legacy output"""
    def __init__(self, model_name='audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim', device='cpu'):
        self.device = torch.device(device)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = EmotionModel.from_pretrained(model_name).to(self.device)
        self.sample_rate = self.processor.feature_extractor.sampling_rate

    def analyze_audio(self, audio: np.ndarray, sample_rate: int) -> tuple:
        logger.debug("[VoiceAnalyzer] --- New Audio Analysis ---")
        try:
            # Step 1: Pre-process audio
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sample_rate != self.sample_rate:
                # This should not happen with proper capture, but is a safeguard
                logger.warning(f"Resampling audio from {sample_rate} to {self.sample_rate}")
                resampler = T.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
                audio = resampler(torch.from_numpy(audio).float()).numpy()
            logger.debug(f"[VoiceAnalyzer] Step 1: Audio pre-processed, shape={audio.shape}")

            # Step 2: Process with Wav2Vec2Processor
            inputs = self.processor(audio, sampling_rate=self.sample_rate, return_tensors='pt', padding=True)
            input_values = inputs['input_values'].to(self.device)
            logger.debug(f"[VoiceAnalyzer] Step 2: Audio processed by Wav2Vec2Processor, shape={input_values.shape}")

            # Step 3: Get model output
            with torch.no_grad():
                _, logits = self.model(input_values)
            logger.debug(f"[VoiceAnalyzer] Step 3: Model output (logits) obtained, shape={logits.shape}")

            # Step 4: Extract valence/arousal
            preds = torch.tanh(logits).cpu().numpy()[0]
            val, aro = float(preds[2]), float(preds[0])
            logger.debug(f"[VoiceAnalyzer] Step 4: Extracted V={val:.3f}, A={aro:.3f}")

            if not (-1.0 <= val <= 1.0 and -1.0 <= aro <= 1.0):
                logger.warning(f"[VoiceAnalyzer] Step 4: Out-of-range valence/arousal from model. Returning neutral.")
                return {'valence': 0.0, 'arousal': 0.0}, 0.0
            
            # Step 5: Calculate confidence based on logit magnitude
            conf = float(torch.sigmoid(torch.norm(logits)).item())
            logger.debug(f"[VoiceAnalyzer] Step 5: Calculated confidence={conf:.3f}")
            
            logger.debug("[VoiceAnalyzer] Analysis successful, returning result.")
            return {'valence': val, 'arousal': aro}, conf

        except Exception as e:
            logger.error(f"[VoiceAnalyzer] --- Analysis failed: {e} ---", exc_info=True)
            return {'valence': 0.0, 'arousal': 0.0}, 0.0


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
