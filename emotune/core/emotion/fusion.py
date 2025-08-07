import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import time

from emotune.utils.logging import get_logger
logger = get_logger()

class EmotionFusion:
    """Advanced weighted fusion of multi-modal emotion estimates with proper uncertainty handling"""
    
    def __init__(self, face_weight: float = 0.7, voice_weight: float = 0.3):
        self.base_face_weight = face_weight
        self.base_voice_weight = voice_weight
        
        # Fusion parameters
        self.min_confidence_threshold = 0.1
        self.max_uncertainty = 0.9
        self.min_uncertainty = 0.1
        
        # Quality metrics
        self._fusion_history = []
        self._max_history_size = 100

        # Dynamic thresholds
        self.face_confidence_threshold = 0.5
        self.voice_confidence_threshold = 0.5
        self.analysis_mode = "fusion"  # fusion, face, or voice

    def set_analysis_mode(self, mode: str):
        """Set the analysis mode."""
        if mode in ["fusion", "face", "voice"]:
            self.analysis_mode = mode
            logger.info(f"Analysis mode set to: {self.analysis_mode}")
        else:
            logger.warning(f"Invalid analysis mode: {mode}")

    def update_thresholds(self, face_threshold: Optional[float] = None, voice_threshold: Optional[float] = None):
        """Update the confidence thresholds for face and voice analysis."""
        if face_threshold is not None:
            self.face_confidence_threshold = face_threshold
        if voice_threshold is not None:
            self.voice_confidence_threshold = voice_threshold
        logger.info(f"Updated fusion thresholds: Face={self.face_confidence_threshold}, Voice={self.voice_confidence_threshold}")

    def fuse_emotions(self, face_data: Optional[Dict] = None,
                     voice_data: Optional[Dict] = None) -> Dict:
        """
        Fuse face and voice emotion estimates with advanced confidence weighting
        Returns: fused emotion estimate with proper uncertainty and covariance
        """
        # Input validation
        if face_data is None and voice_data is None:
            logger.info("No input data for fusion. Returning default emotion.")
            return self._default_emotion()

        # Log incoming data for debugging
        self._log_input_data(face_data, voice_data)

        # Extract and validate emotions
        emotions, weights, confidences = self._extract_emotions_and_weights(face_data, voice_data)
        
        if not emotions:
            logger.info("No valid emotions above confidence threshold. Returning default emotion.")
            return self._default_emotion()

        # Perform weighted fusion
        fused_emotion = self._perform_weighted_fusion(emotions, weights, confidences)
        
        # Calculate uncertainty and covariance
        uncertainty = self._calculate_uncertainty(emotions, confidences, face_data, voice_data)
        covariance = self._calculate_covariance(fused_emotion, face_data, voice_data, uncertainty)
        
        # Build result
        fused_result = {
            'valence': float(fused_emotion['valence']),
            'arousal': float(fused_emotion['arousal']),
            'uncertainty': float(uncertainty),
            'confidence': float(sum(confidences)),
            'sources': {
                'face': face_data is not None and face_data.get('confidence', 0) > self.min_confidence_threshold,
                'voice': voice_data is not None and voice_data.get('confidence', 0) > self.min_confidence_threshold
            },
            'covariance': covariance.tolist()  # Convert to list for JSON serialization
        }

        # Log fusion result
        self._log_fusion_result(fused_result)
        
        # Update history
        self._update_fusion_history(fused_result)
        
        return fused_result

    def _log_input_data(self, face_data: Optional[Dict], voice_data: Optional[Dict]):
        """Log incoming face and voice emotion data"""
        if face_data and 'emotions' in face_data and 'confidence' in face_data:
            emotions = face_data['emotions']
            confidence = face_data['confidence']
            logger.debug(f"Face emotion: V={emotions.get('valence', 0):.3f}, "
                        f"A={emotions.get('arousal', 0):.3f}, C={confidence:.3f}")
        else:
            logger.debug("No valid face data provided")

        if voice_data and 'emotions' in voice_data and 'confidence' in voice_data:
            emotions = voice_data['emotions']
            confidence = voice_data['confidence']
            logger.debug(f"Voice emotion: V={emotions.get('valence', 0):.3f}, "
                        f"A={emotions.get('arousal', 0):.3f}, C={confidence:.3f}")
        else:
            logger.debug("No valid voice data provided")

    def _extract_emotions_and_weights(self, face_data: Optional[Dict], 
                                    voice_data: Optional[Dict]) -> Tuple[List[Dict], List[float], List[float]]:
        """Extract valid emotions and calculate weights"""
        emotions = []
        weights = []
        confidences = []

        if self.analysis_mode == "face" and face_data and face_data.get('confidence', 0) > self.face_confidence_threshold:
            emotions.append(face_data['emotions'])
            weights.append(face_data['confidence'])
            confidences.append(face_data['confidence'])
        elif self.analysis_mode == "voice" and voice_data and voice_data.get('confidence', 0) > self.voice_confidence_threshold:
            emotions.append(voice_data['emotions'])
            weights.append(voice_data['confidence'])
            confidences.append(voice_data['confidence'])
        elif self.analysis_mode == "fusion":
            if face_data and face_data.get('confidence', 0) > self.face_confidence_threshold:
                emotions.append(face_data['emotions'])
                weights.append(face_data['confidence'])
                confidences.append(face_data['confidence'])
            if voice_data and voice_data.get('confidence', 0) > self.voice_confidence_threshold:
                emotions.append(voice_data['emotions'])
                weights.append(voice_data['confidence'])
                confidences.append(voice_data['confidence'])

        if not emotions:
            logger.debug("No valid emotions to fuse.")
            return [], [], []

        return emotions, weights, confidences

    def _is_valid_emotion_value(self, value) -> bool:
        """Check if emotion value is valid"""
        try:
            val = float(value)
            return np.isfinite(val) and -1.0 <= val <= 1.0
        except (ValueError, TypeError):
            return False

    def _perform_weighted_fusion(self, emotions: List[Dict], weights: List[float], 
                               confidences: List[float]) -> Dict:
        """Perform weighted fusion of emotions"""
        if not emotions or not weights:
            return {'valence': 0.0, 'arousal': 0.0}

        # Normalize weights
        weights_array = np.array(weights)
        if np.sum(weights_array) > 0:
            normalized_weights = weights_array / np.sum(weights_array)
        else:
            # Equal weights if all weights are zero
            normalized_weights = np.ones(len(weights)) / len(weights)

        # Weighted fusion
        fused_valence = sum(w * e['valence'] for w, e in zip(normalized_weights, emotions))
        fused_arousal = sum(w * e['arousal'] for w, e in zip(normalized_weights, emotions))

        return {
            'valence': float(fused_valence),
            'arousal': float(fused_arousal)
        }

    def _calculate_uncertainty(self, emotions: List[Dict], confidences: List[float],
                             face_data: Optional[Dict], voice_data: Optional[Dict]) -> float:
        """Calculate uncertainty based on confidence and agreement between modalities"""
        
        if not emotions or not confidences:
            return self.max_uncertainty

        # Base uncertainty from inverse of total confidence
        total_confidence = sum(confidences)
        base_uncertainty = max(0.0, 1.0 - total_confidence)

        # Agreement-based uncertainty (if multiple modalities)
        if len(emotions) > 1:
            # Calculate disagreement between modalities
            valences = [e['valence'] for e in emotions]
            arousals = [e['arousal'] for e in emotions]
            
            valence_disagreement = np.std(valences)
            arousal_disagreement = np.std(arousals)
            
            # Normalize disagreement (max possible std for [-1,1] range is 1.0)
            disagreement = (valence_disagreement + arousal_disagreement) / 2.0
            
            # Combine base uncertainty with disagreement
            uncertainty = base_uncertainty + 0.3 * disagreement
        else:
            uncertainty = base_uncertainty

        # Clamp uncertainty to valid range
        uncertainty = max(self.min_uncertainty, min(self.max_uncertainty, uncertainty))
        
        return float(uncertainty)

    def _calculate_covariance(self, fused_emotion: Dict, face_data: Optional[Dict], 
                            voice_data: Optional[Dict], uncertainty: float) -> np.ndarray:
        """Calculate covariance matrix based on uncertainty and modality agreement"""
        
        # Base variance from uncertainty
        base_variance = uncertainty * 2.0  # Scale uncertainty to variance
        
        # Adjust variance based on number of modalities
        if face_data and voice_data:
            # Two modalities: lower variance due to redundancy
            variance = base_variance * 0.7
        else:
            # Single modality: higher variance due to lack of redundancy
            variance = base_variance * 1.2

        # Add correlation if both modalities are present
        if face_data and voice_data:
            # Estimate correlation based on agreement
            face_emotions = face_data.get('emotions', {})
            voice_emotions = voice_data.get('emotions', {})
            
            valence_diff = abs(face_emotions.get('valence', 0) - voice_emotions.get('valence', 0))
            arousal_diff = abs(face_emotions.get('arousal', 0) - voice_emotions.get('arousal', 0))
            
            # Correlation decreases with disagreement
            correlation = max(0.0, 0.3 - 0.2 * (valence_diff + arousal_diff) / 2.0)
        else:
            correlation = 0.0

        # Build covariance matrix
        covariance = np.array([
            [variance, correlation * variance],
            [correlation * variance, variance]
        ])
        
        return covariance

    def _log_fusion_result(self, fused_result: Dict):
        """Log fusion result with proper formatting"""
        logger.info(f"Fused emotion: V={fused_result['valence']:.3f}, "
                   f"A={fused_result['arousal']:.3f}, "
                   f"U={fused_result['uncertainty']:.3f}, "
                   f"C={fused_result['confidence']:.3f}")

    def _update_fusion_history(self, fused_result: Dict):
        """Update fusion history for quality tracking"""
        self._fusion_history.append({
            'timestamp': time.time(),
            'valence': fused_result['valence'],
            'arousal': fused_result['arousal'],
            'uncertainty': fused_result['uncertainty'],
            'confidence': fused_result['confidence']
        })
        
        # Keep history size manageable
        if len(self._fusion_history) > self._max_history_size:
            self._fusion_history = self._fusion_history[-self._max_history_size:]

    def _default_emotion(self) -> Dict:
        """Default neutral emotion with high uncertainty"""
        return {
            'valence': 0.0,
            'arousal': 0.0,
            'uncertainty': self.max_uncertainty,
            'confidence': 0.0,
            'sources': {'face': False, 'voice': False},
            'covariance': [[1.0, 0.0], [0.0, 1.0]]
        }

    def get_fusion_quality_metrics(self) -> Dict:
        """Get quality metrics from fusion history"""
        if not self._fusion_history:
            return {'average_confidence': 0.0, 'average_uncertainty': 1.0, 'sample_count': 0}
        
        confidences = [entry['confidence'] for entry in self._fusion_history]
        uncertainties = [entry['uncertainty'] for entry in self._fusion_history]
        
        return {
            'average_confidence': float(np.mean(confidences)),
            'average_uncertainty': float(np.mean(uncertainties)),
            'confidence_std': float(np.std(confidences)),
            'uncertainty_std': float(np.std(uncertainties)),
            'sample_count': len(self._fusion_history)
        }

    def reset_history(self):
        """Reset fusion history"""
        self._fusion_history.clear()


# Legacy function for backward compatibility
def estimate_covariance_from_fusion(fused_emotion: Dict, face_data: Optional[Dict], 
                                  voice_data: Optional[Dict]) -> np.ndarray:
    """Legacy covariance estimation function - now uses improved algorithm"""
    # Create a temporary fusion instance to use the new algorithm
    temp_fusion = EmotionFusion()
    
    # Calculate uncertainty using the new method
    emotions, weights, confidences = temp_fusion._extract_emotions_and_weights(face_data, voice_data)
    uncertainty = temp_fusion._calculate_uncertainty(emotions, confidences, face_data, voice_data)
    
    # Calculate covariance using the new method
    covariance = temp_fusion._calculate_covariance(fused_emotion, face_data, voice_data, uncertainty)
    
    return covariance
