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
from emotune.utils.logging import get_logger

# Add emonet to the path for face model
try:
    emonet_path = Path(__file__).parent.parent.parent.parent / 'vendor' / 'emonet'
    sys.path.append(str(emonet_path))
    from emonet.models import EmoNet
except ImportError as e:
    print(f"Error importing EmoNet: {e}")
    print("Please ensure the emonet submodule is available in the vendor directory.")
    sys.exit(1)

# Setup logging (use centralized EmoTune logger)
logger = get_logger()


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
        # Fallback detector (OpenCV Haar)
        try:
            self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.haar.empty():
                self.haar = None
        except Exception:
            self.haar = None

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
            em, conf, _details = self.analyze_frame_detailed(frame, save_dir=None)
            return em, conf
        except Exception as e:
            logger.error(f"[FaceAnalyzer] --- Analysis failed: {e} ---", exc_info=True)
            return {'valence': 0.0, 'arousal': 0.0}, 0.0

    def analyze_frame_detailed(self, frame: np.ndarray, save_dir: str = None) -> tuple:
        """
        Analyze frame and return (emotions, confidence, details).
        details = {
            'detection_method': 'mtcnn' | 'haar' | 'center_crop' | 'none',
            'bbox': [x1, y1, x2, y2] or [x, y, w, h] depending on method,
            'det_conf': float | None,
            'saved_crop_path': str | None
        }
        If save_dir is provided, the detected face crop (or center crop) will be saved there.
        """
        try:
            # Step 1 & 2: Ensure RGB for detection
            if frame.ndim == 3 and frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            img = Image.fromarray(frame_rgb)
            logger.debug(f"[FaceAnalyzer] Input prepared for detection, shape={frame_rgb.shape}")

            detection_method = 'none'
            bbox = None
            det_conf_value = None
            saved_crop_path = None

            # Step 2: Face detection
            boxes, det_probs = self.mtcnn.detect(img)
            if boxes is None or len(boxes) == 0:
                logger.debug("[FaceAnalyzer] Step 2: Detected 0 faces.")
            else:
                logger.debug(f"[FaceAnalyzer] Step 2: Detected {len(boxes)} faces.")

            if boxes is not None and len(boxes) > 0:
                # Step 3: Select face by highest detection prob, fallback to largest area
                areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
                idx_prob = int(np.argmax(det_probs)) if det_probs is not None else None
                idx_area = int(np.argmax(areas)) if len(areas) > 0 else None
                idx = idx_prob if idx_prob is not None else idx_area
                x1, y1, x2, y2 = [int(b) for b in boxes[idx]]
                logger.debug(f"[FaceAnalyzer] Step 3: Largest face selected with box: ({x1}, {y1}, {x2}, {y2})")

                # Step 4: Crop face
                face_crop = frame_rgb[y1:y2, x1:x2]

                if face_crop.size > 0:
                    logger.debug(f"[FaceAnalyzer] Step 4: Cropped face size: {face_crop.shape}")
                    # Save crop if requested
                    if save_dir is not None:
                        try:
                            os.makedirs(save_dir, exist_ok=True)
                            ts = int(time.time() * 1000)
                            crop_path = os.path.join(save_dir, f"face_crop_mtcnn_{ts}.jpg")
                            cv2.imwrite(crop_path, cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
                            saved_crop_path = crop_path
                        except Exception as e:
                            logger.debug(f"[FaceAnalyzer] Failed to save crop: {e}")
                    # Step 5: Preprocess and process with EmoNet
                    face_tensor = self._preprocess_face(face_crop)
                    det_conf_value = float(np.clip(det_probs[idx], 0.0, 1.0)) if det_probs is not None else None
                    detection_method = 'mtcnn'
                    em, conf = self._process_emonet(face_tensor, det_conf_value)
                    logger.debug("[FaceAnalyzer] Analysis successful, returning result.")
                    details = {
                        'detection_method': detection_method,
                        'bbox': [x1, y1, x2, y2],
                        'det_conf': det_conf_value,
                        'saved_crop_path': saved_crop_path
                    }
                    return em, conf, details
                else:
                    logger.warning("[FaceAnalyzer] Step 4: Cropped face has zero size. Proceeding to fallbacks.")
            else:
                logger.info("[FaceAnalyzer] No faces detected by MTCNN. Trying Haar fallback...")

            # --- Haar cascade fallback ---
            if self.haar is not None:
                try:
                    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
                    haar_faces = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
                    if len(haar_faces) > 0:
                        x, y, w, h = max(haar_faces, key=lambda r: r[2] * r[3])
                        logger.debug(f"[FaceAnalyzer] Haar fallback detected face at ({x},{y},{w},{h})")
                        face_crop = frame_rgb[y:y+h, x:x+w]
                        if face_crop.size > 0:
                            if save_dir is not None:
                                try:
                                    os.makedirs(save_dir, exist_ok=True)
                                    ts = int(time.time() * 1000)
                                    crop_path = os.path.join(save_dir, f"face_crop_haar_{ts}.jpg")
                                    cv2.imwrite(crop_path, cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
                                    saved_crop_path = crop_path
                                except Exception as e:
                                    logger.debug(f"[FaceAnalyzer] Failed to save crop: {e}")
                            face_tensor = self._preprocess_face(face_crop)
                            detection_method = 'haar'
                            det_conf_value = 0.4
                            em, conf = self._process_emonet(face_tensor, det_conf=det_conf_value)
                            logger.info("[FaceAnalyzer] Using Haar fallback face crop.")
                            details = {
                                'detection_method': detection_method,
                                'bbox': [int(x), int(y), int(w), int(h)],
                                'det_conf': det_conf_value,
                                'saved_crop_path': saved_crop_path
                            }
                            return em, conf, details
                except Exception as e:
                    logger.debug(f"[FaceAnalyzer] Haar fallback failed: {e}")
            else:
                logger.debug("[FaceAnalyzer] Haar cascade unavailable.")

            # --- Center-crop fallback ---
            try:
                h, w, _ = frame_rgb.shape
                side = int(min(h, w) * 0.6)
                cx, cy = w // 2, h // 2
                x1, y1 = max(0, cx - side // 2), max(0, cy - side // 2)
                x2, y2 = min(w, x1 + side), min(h, y1 + side)
                face_crop = frame_rgb[y1:y2, x1:x2]
                if face_crop.size > 0 and face_crop.shape[0] > 10 and face_crop.shape[1] > 10:
                    if save_dir is not None:
                        try:
                            os.makedirs(save_dir, exist_ok=True)
                            ts = int(time.time() * 1000)
                            crop_path = os.path.join(save_dir, f"face_crop_center_{ts}.jpg")
                            cv2.imwrite(crop_path, cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
                            saved_crop_path = crop_path
                        except Exception as e:
                            logger.debug(f"[FaceAnalyzer] Failed to save crop: {e}")
                    logger.info("[FaceAnalyzer] Using center-crop fallback face region.")
                    face_tensor = self._preprocess_face(face_crop)
                    detection_method = 'center_crop'
                    det_conf_value = 0.2
                    em, conf = self._process_emonet(face_tensor, det_conf=det_conf_value)
                    details = {
                        'detection_method': detection_method,
                        'bbox': [x1, y1, x2, y2],
                        'det_conf': det_conf_value,
                        'saved_crop_path': saved_crop_path
                    }
                    return em, conf, details
            except Exception as e:
                logger.debug(f"[FaceAnalyzer] Center-crop fallback failed: {e}")

            logger.info("[FaceAnalyzer] No face found; returning neutral with zero confidence.")
            details = {
                'detection_method': 'none',
                'bbox': None,
                'det_conf': None,
                'saved_crop_path': None
            }
            return {'valence': 0.0, 'arousal': 0.0}, 0.0, details
        except Exception as e:
            logger.error(f"[FaceAnalyzer] --- Analysis failed: {e} ---", exc_info=True)
            details = {
                'detection_method': 'error',
                'bbox': None,
                'det_conf': None,
                'saved_crop_path': None
            }
            return {'valence': 0.0, 'arousal': 0.0}, 0.0, details

    def _preprocess_face(self, face_crop: np.ndarray) -> torch.Tensor:
        """Resize and normalize face image for EmoNet"""
        face_resized = cv2.resize(face_crop, (256, 256))
        face_tensor = torch.from_numpy(face_resized).permute(2, 0, 1).float().to(self.device) / 255.0
        face_tensor = face_tensor.unsqueeze(0)
        logger.debug(f"[FaceAnalyzer] Preprocessed face tensor, shape={face_tensor.shape}")
        return face_tensor

    def _process_emonet(self, tensor: torch.Tensor, det_conf: float = None) -> tuple:
        """Shared EmoNet processing for both paths. Optionally uses detection confidence to weight final confidence."""
        with torch.no_grad():
            out = self.model(tensor)
            probs = torch.softmax(out['expression'], dim=1).cpu().numpy()[0]
            dist = {self.labels[i]: f"{probs[i]:.3f}" for i in range(len(self.labels))}
            
            em = {
                'valence': float(np.clip(out['valence'].cpu().item(), -1.0, 1.0)),
                'arousal': float(np.clip(out['arousal'].cpu().item(), -1.0, 1.0)),
            }
            # Confidence: combine max expression prob and face detection prob if available
            conf_expr = float(np.max(probs))
            if det_conf is not None:
                conf = float(np.clip(0.7 * conf_expr + 0.3 * float(det_conf), 0.0, 1.0))
            else:
                conf = float(np.clip(conf_expr, 0.0, 1.0))
            
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
            # audeering/msp-dim order is often [arousal, dominance, valence]
            aro = float(np.clip(preds[0], -1.0, 1.0))
            val = float(np.clip(preds[2], -1.0, 1.0))
            logger.debug(f"[VoiceAnalyzer] Step 4: Extracted V={val:.3f}, A={aro:.3f}")

            if not (-1.0 <= val <= 1.0 and -1.0 <= aro <= 1.0):
                logger.warning(f"[VoiceAnalyzer] Step 4: Out-of-range valence/arousal from model. Returning neutral.")
                return {'valence': 0.0, 'arousal': 0.0}, 0.0
            
            # Step 5: Confidence from VA magnitude only (bounded)
            conf = float(np.clip((abs(val) + abs(aro)) / 2.0, 0.0, 1.0))
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
