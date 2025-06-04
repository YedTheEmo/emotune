from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np
import time

from utils.logging import get_logger
logger = get_logger()

@dataclass
class GlobalState:
    """Global state management for EmoTune system"""
    
    # Session state
    session_active: bool = False
    session_id: str = ""
    session_start_time: float = 0.0
    
    # Current system state
    current_emotion: Optional[np.ndarray] = None
    current_emotion_confidence: float = 0.0
    target_emotion: Optional[np.ndarray] = None
    
    # Music state
    current_music_params: Dict[str, float] = field(default_factory=dict)
    music_code: str = ""
    
    # Trajectory state
    trajectory_name: str = ""
    trajectory_progress: float = 0.0
    trajectory_deviation: float = 0.0
    
    # Learning state
    rl_enabled: bool = True
    learning_progress: Dict[str, float] = field(default_factory=dict)
    
    # Performance metrics
    emotion_update_rate: float = 0.0
    music_update_rate: float = 0.0
    system_latency: float = 0.0
    
    def update_emotion_state(self, emotion: np.ndarray, confidence: float):
        """Update current emotion state"""
        self.current_emotion = emotion.copy() if emotion is not None else None
        self.current_emotion_confidence = confidence
    
    def update_music_state(self, params: Dict[str, float], code: str):
        """Update current music state"""
        self.current_music_params = params.copy()
        self.music_code = code
    
    def update_trajectory_state(self, target: np.ndarray, progress: float, deviation: float):
        """Update trajectory state"""
        self.target_emotion = target.copy() if target is not None else None
        self.trajectory_progress = progress
        self.trajectory_deviation = deviation
    
    def get_session_duration(self) -> float:
        """Get current session duration"""
        if not self.session_active:
            return 0.0
        return time.time() - self.session_start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization"""

        def safe_tolist(obj):
            try:
                return obj.tolist()
            except AttributeError:
                return obj

        if self.current_emotion is not None:
            logger.info(f"[to_dict] current_emotion type: {type(self.current_emotion)}")
        if self.target_emotion is not None:
            logger.info(f"[to_dict] target_emotion type: {type(self.target_emotion)}")

        return {
            'session_active': self.session_active,
            'session_id': self.session_id,
            'session_duration': self.get_session_duration(),
            'current_emotion': safe_tolist(self.current_emotion),
            'current_emotion_confidence': self.current_emotion_confidence,
            'target_emotion': safe_tolist(self.target_emotion),
            'current_music_params': self.current_music_params,
            'trajectory_name': self.trajectory_name,
            'trajectory_progress': self.trajectory_progress,
            'trajectory_deviation': self.trajectory_deviation,
            'rl_enabled': self.rl_enabled,
            'performance_metrics': {
                'emotion_update_rate': self.emotion_update_rate,
                'music_update_rate': self.music_update_rate,
                'system_latency': self.system_latency
            }
        }
