import time
from collections import deque
from typing import Dict, List, Optional, Any
import numpy as np
from typing import TypedDict

from emotune.utils.logging import get_logger
logger = get_logger()

class EmotionMean(TypedDict):
    valence: float
    arousal: float

class EmotionDistribution(TypedDict, total=False):
    mean: EmotionMean
    covariance: list
    uncertainty_trace: float
    timestamp: float

class EmotionState:
    """Manages current emotion state and history with improved data validation"""
    
    def __init__(self, history_length: int = 100):
        self.history_length = history_length
        
        # Current state
        self.current_emotion = {
            'mean': {'valence': 0.0, 'arousal': 0.0},
            'covariance': [[0.5, 0.0], [0.0, 0.5]],
            'uncertainty_trace': 1.0,
            'timestamp': time.time(),
            'confidence': 0.5
        }
        
        # History storage
        self.emotion_history = deque(maxlen=history_length)
        self.raw_observations = deque(maxlen=history_length)
        
    def _is_valid_emotion(self, emotion_dist: Any) -> bool:
        """Check if an emotion distribution is valid with improved validation"""
        if not isinstance(emotion_dist, dict):
            return False
            
        # Check for required fields
        required_fields = ['mean', 'covariance']
        if not all(field in emotion_dist for field in required_fields):
            return False
            
        mean = emotion_dist.get('mean')
        if not (isinstance(mean, dict) and 'valence' in mean and 'arousal' in mean):
            return False
            
        # Validate valence and arousal values
        try:
            v = float(mean['valence'])
            a = float(mean['arousal'])
            if not (-1.0 <= v <= 1.0 and -1.0 <= a <= 1.0):
                return False
            if not (np.isfinite(v) and np.isfinite(a)):
                return False
        except (ValueError, TypeError):
            return False
            
        # Validate covariance
        cov = emotion_dist.get('covariance')
        if not self._is_valid_covariance(cov):
            return False
            
        # Validate timestamp
        ts = emotion_dist.get('timestamp')
        if ts is None or not np.isfinite(ts):
            return False
            
        return True

    def _is_valid_covariance(self, cov: Any) -> bool:
        """Check if covariance is valid (2x2 matrix)"""
        if cov is None:
            return False
            
        # Handle numpy arrays
        if isinstance(cov, np.ndarray):
            if cov.shape != (2, 2):
                return False
            return np.all(np.isfinite(cov))
            
        # Handle lists
        if isinstance(cov, list):
            if len(cov) != 2:
                return False
            for row in cov:
                if not isinstance(row, list) or len(row) != 2:
                    return False
                for val in row:
                    try:
                        if not np.isfinite(float(val)):
                            return False
                    except (ValueError, TypeError):
                        return False
            return True
            
        return False

    def _normalize_covariance(self, cov: Any) -> List[List[float]]:
        """Normalize covariance to 2x2 list format"""
        if cov is None:
            return [[0.5, 0.0], [0.0, 0.5]]
            
        # Handle numpy arrays
        if isinstance(cov, np.ndarray):
            if cov.shape == (2, 2):
                return cov.tolist()
            else:
                logger.warning(f"Invalid covariance shape: {cov.shape}, using default")
                return [[0.5, 0.0], [0.0, 0.5]]
                
        # Handle lists
        if isinstance(cov, list):
            if len(cov) == 2 and all(isinstance(row, list) and len(row) == 2 for row in cov):
                try:
                    return [[float(cov[0][0]), float(cov[0][1])], 
                           [float(cov[1][0]), float(cov[1][1])]]
                except (ValueError, TypeError):
                    logger.warning("Invalid covariance values, using default")
                    return [[0.5, 0.0], [0.0, 0.5]]
                    
        # Default fallback
        logger.warning(f"Unknown covariance format: {type(cov)}, using default")
        return [[0.5, 0.0], [0.0, 0.5]]

    def update_emotion(self, emotion_dist: Any):
        """Update current emotion state with improved validation and normalization"""
        logger.debug(f"[EmotionState] update_emotion called with: {emotion_dist}")
        
        # Type and structure validation
        if not isinstance(emotion_dist, dict):
            logger.warning("Attempted to update emotion with non-dict object. Ignored.")
            return
            
        # Extract and validate mean
        mean = emotion_dist.get('mean')
        if mean is None:
            logger.warning("Emotion update missing 'mean'. Ignored.")
            return
            
        if not (isinstance(mean, dict) and 'valence' in mean and 'arousal' in mean):
            logger.warning("Emotion update 'mean' missing 'valence' or 'arousal'. Ignored.")
            return
            
        # Validate and convert mean values
        try:
            v = float(mean['valence'])
            a = float(mean['arousal'])
            if not (np.isfinite(v) and np.isfinite(a)):
                logger.warning("Emotion update contains non-finite values. Ignored.")
                return
            if not (-1.0 <= v <= 1.0 and -1.0 <= a <= 1.0):
                logger.warning("Emotion update valence/arousal out of range [-1,1]. Ignored.")
                return
        except (ValueError, TypeError) as e:
            logger.warning(f"Emotion update value error: {e}. Ignored.")
            return
            
        # Normalize covariance
        cov = emotion_dist.get('covariance')
        normalized_cov = self._normalize_covariance(cov)
        
        # Ensure timestamp
        timestamp = emotion_dist.get('timestamp')
        if timestamp is None or not np.isfinite(timestamp):
            timestamp = time.time()
            
        # Create normalized emotion distribution
        normalized_emotion = {
            'mean': {
                'valence': float(v),
                'arousal': float(a)
            },
            'covariance': normalized_cov,
            'uncertainty_trace': float(emotion_dist.get('uncertainty_trace', 1.0)),
            'timestamp': float(timestamp),
            'confidence': float(emotion_dist.get('confidence', 0.5))
        }
        
        # Update current emotion
        self.current_emotion = normalized_emotion.copy()
        
        # Add to history
        self.emotion_history.append(normalized_emotion.copy())
        
        logger.info(f"[EmotionState] Updated emotion. History length: {len(self.emotion_history)}")
        logger.debug(f"[EmotionState] Latest emotion: {normalized_emotion}")

    def add_raw_observation(self, observation: Dict):
        """Add raw observation to history with timestamp"""
        observation['timestamp'] = time.time()
        self.raw_observations.append(observation)
        
    def get_current_emotion(self) -> Dict:
        """Get current emotion distribution"""
        return self.current_emotion.copy()
    
    def get_emotion_trajectory(self, time_window: float = 60.0) -> List[Dict]:
        """Get emotion trajectory within time window with improved error handling"""
        try:
            current_time = time.time()
            cutoff_time = current_time - time_window
            trajectory = [
                emotion for emotion in self.emotion_history
                if emotion['timestamp'] >= cutoff_time
            ]
            
            if not trajectory and len(self.emotion_history) > 0:
                # Fallback: return the latest emotion if nothing in window
                trajectory = [self.emotion_history[-1]]
                
            logger.debug(f"[EmotionState] get_emotion_trajectory returned {len(trajectory)} items.")
            return trajectory
            
        except Exception as e:
            logger.error(f"Error getting emotion trajectory: {e}")
            return []
    
    def get_emotion_statistics(self, time_window: float = 60.0) -> Dict:
        """Get emotion statistics over time window with improved error handling"""
        try:
            trajectory = self.get_emotion_trajectory(time_window)
            
            if not trajectory:
                return {
                    'mean_valence': 0.0,
                    'mean_arousal': 0.0,
                    'std_valence': 0.0,
                    'std_arousal': 0.0,
                    'trajectory_length': 0,
                    'time_span': 0.0
                }
            
            # Extract valence and arousal values
            valences = []
            arousals = []
            timestamps = []
            
            for emotion in trajectory:
                try:
                    mean = emotion.get('mean', {})
                    valence = mean.get('valence', 0.0)
                    arousal = mean.get('arousal', 0.0)
                    timestamp = emotion.get('timestamp', 0.0)
                    
                    if np.isfinite(valence) and np.isfinite(arousal) and np.isfinite(timestamp):
                        valences.append(float(valence))
                        arousals.append(float(arousal))
                        timestamps.append(float(timestamp))
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid emotion data in trajectory: {e}")
                    continue
            
            if not valences:
                return {
                    'mean_valence': 0.0,
                    'mean_arousal': 0.0,
                    'std_valence': 0.0,
                    'std_arousal': 0.0,
                    'trajectory_length': 0,
                    'time_span': 0.0
                }
            
            # Calculate statistics
            mean_valence = float(np.mean(valences))
            mean_arousal = float(np.mean(arousals))
            std_valence = float(np.std(valences))
            std_arousal = float(np.std(arousals))
            
            time_span = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0.0
            
            return {
                'mean_valence': mean_valence,
                'mean_arousal': mean_arousal,
                'std_valence': std_valence,
                'std_arousal': std_arousal,
                'trajectory_length': len(valences),
                'time_span': time_span
            }
            
        except Exception as e:
            logger.error(f"Error calculating emotion statistics: {e}")
            return {
                'mean_valence': 0.0,
                'mean_arousal': 0.0,
                'std_valence': 0.0,
                'std_arousal': 0.0,
                'trajectory_length': 0,
                'time_span': 0.0
            }

    def clear_history(self):
        """Clear emotion history"""
        self.emotion_history.clear()
        self.raw_observations.clear()
        logger.info("Emotion history cleared")

    def reset(self):
        """Reset current emotion state and clear history"""
        self.current_emotion = {
            'mean': {'valence': 0.0, 'arousal': 0.0},
            'covariance': [[0.5, 0.0], [0.0, 0.5]],
            'uncertainty_trace': 1.0,
            'timestamp': time.time(),
            'confidence': 0.5
        }
        self.emotion_history.clear()
        self.raw_observations.clear()
        logger.info("Emotion state reset")

    def get_stability_metric(self, time_window: float = 60.0) -> float:
        """
        Calculate an emotional stability metric based on variance of valence and arousal
        over the recent time window. Lower values indicate higher stability.
        """
        try:
            stats = self.get_emotion_statistics(time_window)
            # Combine standard deviations as inverse stability
            stability = 1.0 / (1e-6 + stats['std_valence'] + stats['std_arousal'])
            return float(stability)
        except Exception as e:
            logger.error(f"Error calculating stability metric: {e}")
            return 1.0

    def get_update_count(self) -> int:
        """Return the number of emotion updates recorded so far"""
        return len(self.emotion_history)

    def is_valid(self) -> bool:
        """Return True if the current emotion state is valid"""
        return self._is_valid_emotion(self.current_emotion)

    def get_latest_emotion_values(self) -> Dict[str, float]:
        """Get latest valence and arousal values for easy access"""
        try:
            mean = self.current_emotion.get('mean', {})
            return {
                'valence': float(mean.get('valence', 0.0)),
                'arousal': float(mean.get('arousal', 0.0)),
                'confidence': 1.0 - min(1.0, self.current_emotion.get('uncertainty_trace', 0.5) / 2.0)
            }
        except Exception as e:
            logger.error(f"Error getting latest emotion values: {e}")
            return {'valence': 0.0, 'arousal': 0.0, 'confidence': 0.5}
