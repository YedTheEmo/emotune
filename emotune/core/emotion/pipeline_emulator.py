import time
from typing import Dict, Optional, Any
import numpy as np

from .fusion import EmotionFusion
from .kalman_filter import KalmanEmotionFilter


class PipelineEmulator:
    """
    Lightweight emulator of the capture -> analysis -> fusion -> filter pipeline.
    This module avoids any hardware or heavy model dependencies and is intended
    for unit testing the downstream logic using simple inputs.
    
    Inputs:
      - face: dict with keys {valence, arousal, confidence} or nested like
        {'emotions': {'valence': v, 'arousal': a}, 'confidence': c}
      - voice: same as face
    
    Behavior:
      - Normalizes inputs to EmotionFusion format
      - Runs EmotionFusion (respects analysis_mode, thresholds, min_conf, fallbacks)
      - Wraps result to 'mean' format if needed
      - Applies KalmanEmotionFilter with the same rule as SessionManager:
        predict-only if no sources or confidence < 0.2; otherwise update
    Returns dict with keys: raw, fused, filtered
    """

    def __init__(self,
                 face_weight: float = 0.7,
                 voice_weight: float = 0.3):
        self.fusion = EmotionFusion(face_weight=face_weight, voice_weight=voice_weight)
        self.kalman = KalmanEmotionFilter()

    # --- Configuration passthroughs ---
    def set_analysis_mode(self, mode: str):
        self.fusion.set_analysis_mode(mode)

    def set_confidence_thresholds(self, face: Optional[float] = None, voice: Optional[float] = None):
        self.fusion.update_thresholds(face_threshold=face, voice_threshold=voice)

    def set_fusion_options(self, allow_fallback: Optional[bool] = None, fusion_min_conf: Optional[float] = None):
        self.fusion.update_options(allow_fallback=allow_fallback, fusion_min_conf=fusion_min_conf)

    # --- Emulation ---
    def step(self,
             face: Optional[Dict[str, Any]] = None,
             voice: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run one emulation step given optional face/voice inputs.
        Returns a dict with raw (normalized inputs), fused (pre-filter), and filtered outputs.
        """
        # Normalize inputs to EmotionFusion modality format
        face_norm = self._normalize_modality(face)
        voice_norm = self._normalize_modality(voice)

        fused = self.fusion.fuse_emotions(face_norm, voice_norm)
        fused = self._wrap_mean_if_needed(fused, face_norm, voice_norm)

        # Apply Kalman using SessionManager rule
        valence = fused['mean']['valence']
        arousal = fused['mean']['arousal']
        confidence = float(np.clip(fused.get('confidence', 0.5), 0.0, 1.0))
        sources = fused.get('sources', {})
        has_observation = bool(sources.get('face') or sources.get('voice'))

        if (not has_observation) or (confidence < 0.2):
            state, cov = self.kalman.predict()
        else:
            state, cov = self.kalman.update({
                'valence': valence,
                'arousal': arousal,
                'confidence': confidence,
                'uncertainty': 1.0 - confidence
            })

        filtered_valence = float(state[0])
        filtered_arousal = float(state[1])
        cov_2x2 = cov[:2, :2] if cov is not None and cov.shape[0] >= 2 and cov.shape[1] >= 2 else np.array([[0.5, 0.0], [0.0, 0.5]])

        filtered = {
            'mean': {'valence': filtered_valence, 'arousal': filtered_arousal},
            'covariance': cov_2x2.tolist(),
            'timestamp': fused.get('timestamp', time.time()),
            'confidence': confidence,
            'sources': sources
        }

        return {
            'raw': {
                'face': face_norm,
                'voice': voice_norm
            },
            'fused': fused,
            'filtered': filtered
        }

    # --- Helpers ---
    def _normalize_modality(self, data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if data is None:
            return None
        # If already in nested format
        if 'emotions' in data and isinstance(data['emotions'], dict):
            v = data['emotions'].get('valence')
            a = data['emotions'].get('arousal')
            c = data.get('confidence', 0.0)
            return {'emotions': {'valence': self._clip(v), 'arousal': self._clip(a)}, 'confidence': float(np.clip(c, 0.0, 1.0))}
        # Flat format: {valence, arousal, confidence}
        v = data.get('valence')
        a = data.get('arousal')
        c = data.get('confidence', 0.0)
        return {'emotions': {'valence': self._clip(v), 'arousal': self._clip(a)}, 'confidence': float(np.clip(c, 0.0, 1.0))}

    def _wrap_mean_if_needed(self, fused: Dict[str, Any], face_data: Optional[Dict[str, Any]], voice_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        # Ensure timestamp and raw attachments
        out = dict(fused)
        out['timestamp'] = out.get('timestamp', time.time())
        if face_data is not None:
            out['face'] = face_data
        if voice_data is not None:
            out['voice'] = voice_data

        if 'mean' in out:
            # Ensure clipping and preserve uncertainty
            out['mean'] = {
                'valence': self._clip(out['mean'].get('valence', 0.0)),
                'arousal': self._clip(out['mean'].get('arousal', 0.0))
            }
            out['confidence'] = float(np.clip(out.get('confidence', 0.5), 0.0, 1.0))
            if 'uncertainty' not in out and 'uncertainty' in fused:
                out['uncertainty'] = fused['uncertainty']
            return out

        # Wrap direct vals into mean
        if 'valence' in out and 'arousal' in out:
            wrapped = {
                'mean': {
                    'valence': self._clip(out['valence']),
                    'arousal': self._clip(out['arousal'])
                },
                'confidence': float(np.clip(out.get('confidence', 0.5), 0.0, 1.0)),
                'timestamp': out['timestamp'],
                'sources': out.get('sources', {}),
                'face': out.get('face'),
                'voice': out.get('voice')
            }
            if 'uncertainty' in out:
                wrapped['uncertainty'] = out['uncertainty']
            return wrapped

        # Fallback neutral
        return {
            'mean': {'valence': 0.0, 'arousal': 0.0},
            'confidence': 0.0,
            'timestamp': out['timestamp'],
            'sources': out.get('sources', {}),
            'face': out.get('face'),
            'voice': out.get('voice')
        }

    @staticmethod
    def _clip(x: Any) -> float:
        try:
            return float(np.clip(float(x), -1.0, 1.0))
        except Exception:
            return 0.0 