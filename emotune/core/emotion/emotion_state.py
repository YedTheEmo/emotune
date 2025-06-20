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
    """Manages current emotion state and history"""
    
    def __init__(self, history_length: int = 100):
        self.history_length = history_length
        
        # Current state
        self.current_emotion = {
            'mean': {'valence': 0.0, 'arousal': 0.0},
            'covariance': [[0.5, 0.0], [0.0, 0.5]],
            'uncertainty_trace': 1.0,
            'timestamp': time.time()
        }
        
        # History storage
        self.emotion_history = deque(maxlen=history_length)
        self.raw_observations = deque(maxlen=history_length)
        
    def _is_valid_emotion(self, emotion_dist: Any) -> bool:
        """Check if an emotion distribution is valid (valence/arousal in [-1,1], finite, has timestamp)."""
        if not isinstance(emotion_dist, dict):
            return False
        mean = emotion_dist.get('mean')
        if not (isinstance(mean, dict) and 'valence' in mean and 'arousal' in mean):
            return False
        try:
            v = float(mean['valence'])
            a = float(mean['arousal'])
            if not (-1.0 <= v <= 1.0 and -1.0 <= a <= 1.0):
                return False
            if not (np.isfinite(v) and np.isfinite(a)):
                return False
        except Exception:
            return False
        ts = emotion_dist.get('timestamp')
        if ts is None or not np.isfinite(ts):
            return False
        return True

    def update_emotion(self, emotion_dist: Any):
        """Update current emotion state with type, structure, and value validation."""
        logger.debug(f"[EmotionState] update_emotion called with: {emotion_dist}")
        # Type and structure validation
        if not isinstance(emotion_dist, dict):
            logger.warning("Attempted to update emotion with non-dict object. Ignored.")
            return
        mean = emotion_dist.get('mean')
        cov = emotion_dist.get('covariance')
        if mean is None or cov is None:
            logger.warning("Emotion update missing 'mean' or 'covariance'. Ignored.")
            return
        if not (isinstance(mean, dict) and 'valence' in mean and 'arousal' in mean):
            logger.warning("Emotion update 'mean' missing 'valence' or 'arousal'. Ignored.")
            return
        # Type checks for mean values
        try:
            v = float(mean['valence'])
            a = float(mean['arousal'])
            if not (np.isfinite(v) and np.isfinite(a)):
                logger.warning("Emotion update contains non-finite values. Ignored.")
                return
            if not (-1.0 <= v <= 1.0 and -1.0 <= a <= 1.0):
                logger.warning("Emotion update valence/arousal out of range [-1,1]. Ignored.")
                return
        except Exception as e:
            logger.warning(f"Emotion update value error: {e}. Ignored.")
            return
        # Type check for covariance
        if not (isinstance(cov, list) and len(cov) == 2 and all(isinstance(row, list) and len(row) == 2 for row in cov)):
            logger.warning("Emotion update 'covariance' is not a 2x2 list. Ignored.")
            return
        # Ensure timestamp
        if 'timestamp' not in emotion_dist or not np.isfinite(emotion_dist['timestamp']):
            emotion_dist['timestamp'] = time.time()
        self.current_emotion = emotion_dist
        self.emotion_history.append(emotion_dist.copy())
        logger.info(f"[EmotionState] Updated emotion. History length: {len(self.emotion_history)}. Latest: {emotion_dist}")
        logger.info(f"[EmotionState] emotion_history length after append: {len(self.emotion_history)}")

    def add_raw_observation(self, observation: Dict):
        """Add raw observation to history"""
        observation['timestamp'] = time.time()
        self.raw_observations.append(observation)
        
    def get_current_emotion(self) -> Dict:
        """Get current emotion distribution"""
        return self.current_emotion.copy()
    
    def get_emotion_trajectory(self, time_window: float = 60.0) -> List[Dict]:
        """Get emotion trajectory within time window. Always return at least the latest emotion if available."""
        current_time = time.time()
        cutoff_time = current_time - time_window
        trajectory = [
            emotion for emotion in self.emotion_history
            if emotion['timestamp'] >= cutoff_time
        ]
        if not trajectory and len(self.emotion_history) > 0:
            # Fallback: return the latest emotion if nothing in window
            trajectory = [self.emotion_history[-1]]
        logger.info(f"[EmotionState] get_emotion_trajectory returned {len(trajectory)} items.")
        logger.info(f"[EmotionState] get_emotion_trajectory returning {len(trajectory)} items")
        return trajectory
    
    def get_emotion_statistics(self, time_window: float = 60.0) -> Dict:
        """Get emotion statistics over time window"""
        trajectory = self.get_emotion_trajectory(time_window)
        
        if not trajectory:
            return {
                'mean_valence': 0.0,
                'mean_arousal': 0.0,
                'std_valence': 0.0,
                'std_arousal': 0.0,
                'trajectory_length': 0
            }
        
        valences = [e['mean']['valence'] for e in trajectory]
        arousals = [e['mean']['arousal'] for e in trajectory]
        
        return {
            'mean_valence': float(np.mean(valences)),
            'mean_arousal': float(np.mean(arousals)),
            'std_valence': float(np.std(valences)),
            'std_arousal': float(np.std(arousals)),
            'trajectory_length': len(trajectory),
            'time_span': trajectory[-1]['timestamp'] - trajectory[0]['timestamp'] if len(trajectory) > 1 else 0.0
        }

    def clear_history(self):
        """Clear emotion history"""
        self.emotion_history.clear()
        self.raw_observations.clear()
        logger.info("Emotion history cleared")

    def reset(self):
        """Reset current emotion state and clear history."""
        self.current_emotion = {
            'mean': {'valence': 0.0, 'arousal': 0.0},
            'covariance': [[0.5, 0.0], [0.0, 0.5]],
            'uncertainty_trace': 1.0,
            'timestamp': time.time()
        }
        self.emotion_history.clear()
        self.raw_observations.clear()

    def get_stability_metric(self, time_window: float = 60.0) -> float:
        """
        Calculate an emotional stability metric based on variance of valence and arousal
        over the recent time window.
        Lower values indicate higher stability.
        """
        stats = self.get_emotion_statistics(time_window)
        # Combine standard deviations as inverse stability
        stability = 1.0 / (1e-6 + stats['std_valence'] + stats['std_arousal'])
        return stability

    def get_update_count(self) -> int:
        """Return the number of emotion updates recorded so far"""
        return len(self.emotion_history)

    def is_valid(self) -> bool:
        """Return True if the current emotion state is valid."""
        return self._is_valid_emotion(self.current_emotion)
