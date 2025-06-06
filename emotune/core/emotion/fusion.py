import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from emotune.utils.logging import get_logger
logger = get_logger()

class EmotionFusion:
    """Weighted fusion of multi-modal emotion estimates"""
    
    def __init__(self, face_weight: float = 0.6, voice_weight: float = 0.4):
        self.base_face_weight = face_weight
        self.base_voice_weight = voice_weight
   
    def fuse_emotions(self, face_data: Optional[Dict] = None,
                     voice_data: Optional[Dict] = None) -> Dict:
        """
        Fuse face and voice emotion estimates with confidence weighting
        Returns: fused emotion estimate with uncertainty
        """
        if face_data is None and voice_data is None:
            logger.info("No input data for fusion. Returning default emotion.")
            return self._default_emotion()

        # Log incoming face and voice emotion data
        if face_data:
            logger.info(f"Face emotion detected: Valence={face_data['emotions']['valence']:.3f}, "
                        f"Arousal={face_data['emotions']['arousal']:.3f}, "
                        f"Confidence={face_data['confidence']:.3f}")
        else:
            logger.info("No face data provided.")

        if voice_data:
            logger.info(f"Voice emotion detected: Valence={voice_data['emotions']['valence']:.3f}, "
                        f"Arousal={voice_data['emotions']['arousal']:.3f}, "
                        f"Confidence={voice_data['confidence']:.3f}")
        else:
            logger.info("No voice data provided.")

        # Extract emotions and confidences
        emotions = []
        weights = []

        if face_data and face_data['confidence'] > 0.1:
            emotions.append({
                'valence': face_data['emotions']['valence'],
                'arousal': face_data['emotions']['arousal']
            })
            weights.append(face_data['confidence'] * self.base_face_weight)

        if voice_data and voice_data['confidence'] > 0.1:
            emotions.append({
                'valence': voice_data['emotions']['valence'],
                'arousal': voice_data['emotions']['arousal']
            })
            weights.append(voice_data['confidence'] * self.base_voice_weight)

        if not emotions:
            logger.info("No valid emotions above confidence threshold. Returning default emotion.")
            return self._default_emotion()

        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)

        # Weighted fusion
        fused_valence = sum(w * e['valence'] for w, e in zip(weights, emotions))
        logger.info("DEBUG unpacking at line: fused_arousal = sum(w * e['arousal'] for w, e in zip(weights, emotions))")
        fused_arousal = sum(w * e['arousal'] for w, e in zip(weights, emotions))

        # Calculate uncertainty (inverse of total confidence)
        total_confidence = np.sum([
            (face_data['confidence'] if face_data else 0.0),
            (voice_data['confidence'] if voice_data else 0.0)
        ])

        # Uncertainty increases when confidences are low or disagree
        if len(emotions) > 1:
            val_disagreement = abs(emotions[0]['valence'] - emotions[1]['valence'])
            aro_disagreement = abs(emotions[0]['arousal'] - emotions[1]['arousal'])
            disagreement = (val_disagreement + aro_disagreement) / 2.0
            uncertainty = (1.0 - total_confidence) + 0.5 * disagreement
        else:
            uncertainty = 1.0 - total_confidence

        uncertainty = max(0.1, min(0.9, uncertainty))  # Clamp

        fused_result = {
            'valence': float(fused_valence),
            'arousal': float(fused_arousal),
            'uncertainty': float(uncertainty),
            'confidence': float(total_confidence),
            'sources': {
                'face': face_data is not None,
                'voice': voice_data is not None
            }
        }
        fused_result['covariance'] = estimate_covariance_from_fusion(
        fused_result, face_data, voice_data
        )

        # Log fusion result
        logger.info(f"Fused emotion result: Valence={fused_result['valence']:.3f}, "
                    f"Arousal={fused_result['arousal']:.3f}, "
                    f"Uncertainty={fused_result['uncertainty']:.3f}, "
                    f"Confidence={fused_result['confidence']:.3f}")

        return fused_result


    def _default_emotion(self) -> Dict:
        """Default neutral emotion with high uncertainty"""
        return {
            'valence': 0.0,
            'arousal': 0.0,
            'uncertainty': 0.8,
            'confidence': 0.2,
            'sources': {'face': False, 'voice': False}
        }


def estimate_covariance_from_fusion(fused_emotion: Dict, face_data: Optional[Dict], voice_data: Optional[Dict]) -> np.ndarray:
    """Estimate a diagonal covariance matrix based on modality confidence and disagreement."""
    val_var, aro_var = 0.1, 0.1  # Default low variance (high confidence)

    conf_face = face_data['confidence'] if face_data else 0.0
    conf_voice = voice_data['confidence'] if voice_data else 0.0
    total_conf = conf_face + conf_voice

    # Base inverse confidence (lower confidence = higher variance)
    if total_conf > 0:
        val_var = aro_var = min(1.0, 1.0 / total_conf)  # e.g., 0.5 conf => var = 2.0
    else:
        val_var = aro_var = 1.0  # No confidence = maximum uncertainty

    # Optional: inflate variance if disagreement is high
    if face_data and voice_data:
        val_diff = abs(face_data['emotions']['valence'] - voice_data['emotions']['valence'])
        aro_diff = abs(face_data['emotions']['arousal'] - voice_data['emotions']['arousal'])
        disagreement = (val_diff + aro_diff) / 2.0
        disagreement = min(disagreement, 1.0)

        inflation = 1.0 + 2.0 * disagreement  # more disagreement = more variance
        val_var *= inflation
        aro_var *= inflation

    cov = np.array([[val_var, 0.0], [0.0, aro_var]])
    return cov
