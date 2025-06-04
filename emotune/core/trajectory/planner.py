import time
from typing import Dict, Optional, Callable, Tuple
from enum import Enum
from .library import TrajectoryLibrary, TrajectoryType
from .dtw_matcher import DTWMatcher
import logging

from utils.logging import get_logger
logger = get_logger()

class TrajectoryStatus(Enum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    COMPLETED = "completed"
    ADAPTING = "adapting"

class TrajectoryPlanner:
    """Manages therapeutic trajectory planning and execution"""
    
    def __init__(self):
        self.library = TrajectoryLibrary()
        self.dtw_matcher = DTWMatcher(window_size=10)
        
        # Current trajectory state
        self.current_trajectory = None
        self.current_type = None
        self.start_time = None
        self.duration = None
        self.status = TrajectoryStatus.INACTIVE
        
        # Adaptation parameters
        self.deviation_threshold = 0.3
        self.adaptation_enabled = True
        
    def start_trajectory(self, trajectory_type: TrajectoryType,
                        duration: float = 300.0,
                        start_state: Dict = None,
                        target_state: Dict = None) -> bool:
        """Start a new therapeutic trajectory"""
        try:
            self.current_trajectory = self.library.get_trajectory(
                trajectory_type, duration, start_state, target_state
            )
            self.current_type = trajectory_type
            self.start_time = time.time()
            self.duration = duration
            self.status = TrajectoryStatus.ACTIVE
            
            logger.info(f"Started trajectory: {trajectory_type.value} for {duration}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start trajectory: {e}")
            return False
    
    def stop_trajectory(self):
        """Stop current trajectory"""
        if self.status != TrajectoryStatus.INACTIVE:
            logger.info(f"Stopped trajectory: {self.current_type.value if self.current_type else 'Unknown'}")
            
        self.current_trajectory = None
        self.current_type = None
        self.start_time = None
        self.duration = None
        self.status = TrajectoryStatus.INACTIVE
    
    def get_current_target(self) -> Optional[Tuple[float, float]]:
        """Get current trajectory target point"""
        if not self._is_active():
            return None
            
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        if elapsed >= self.duration:
            self.status = TrajectoryStatus.COMPLETED
            return self.current_trajectory(self.duration)
            
        return self.current_trajectory(elapsed)
    
    def evaluate_trajectory_adherence(self, emotion_trajectory: list) -> Dict:
        """Evaluate how well actual trajectory matches target"""
        if not self._is_active() or not emotion_trajectory:
            return {
                'deviation': 0.0,
                'adherence_score': 1.0,
                'needs_adaptation': False
            }
        
        # Compute DTW deviation
        deviation = self.dtw_matcher.compute_trajectory_deviation(
            emotion_trajectory, self.current_trajectory, self.start_time
        )


        
        adherence_score = 1.0 - deviation
        needs_adaptation = (deviation > self.deviation_threshold and 
                          self.adaptation_enabled)
        
        return {
            'deviation': float(deviation),
            'adherence_score': float(adherence_score),
            'needs_adaptation': needs_adaptation,
            'trajectory_type': self.current_type.value if self.current_type else None,
            'elapsed_time': time.time() - self.start_time if self.start_time else 0.0,
            'progress': min(1.0, (time.time() - self.start_time) / self.duration) if self._is_active() else 0.0
        }
    
    def get_trajectory_info(self) -> Dict:
        """Get current trajectory information"""
        if not self._is_active():
            return {
                'active': False,
                'type': None,
                'progress': 0.0,
                'target': None,
                'status': self.status.value
            }
        
        current_time = time.time()
        elapsed = current_time - self.start_time
        progress = min(1.0, elapsed / self.duration)
        target = self.get_current_target()
        
        return {
            'active': True,
            'type': self.current_type.value,
            'progress': float(progress),
            'target': {
                'valence': target[0],
                'arousal': target[1]
            } if target else None,
            'elapsed_time': float(elapsed),
            'total_duration': float(self.duration),
            'status': self.status.value
        }
    
    def _is_active(self) -> bool:
        """Check if trajectory is currently active"""
        return (self.status in [TrajectoryStatus.ACTIVE, TrajectoryStatus.ADAPTING] and
                self.current_trajectory is not None)
