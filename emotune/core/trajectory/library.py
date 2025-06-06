import numpy as np
from typing import Dict, List, Callable, Tuple
from enum import Enum
import logging

from emotune.utils.logging import get_logger
logger = get_logger()

class TrajectoryType(Enum):
    CALM_DOWN = "calm_down"
    ENERGIZE = "energize" 
    FOCUS = "focus"
    RELAX = "relax"
    MOOD_LIFT = "mood_lift"
    STABILIZE = "stabilize"

class TrajectoryLibrary:
    """Library of therapeutic trajectory templates"""
    
    def __init__(self):
        self.trajectories = {
            TrajectoryType.CALM_DOWN: self._calm_down_trajectory,
            TrajectoryType.ENERGIZE: self._energize_trajectory,
            TrajectoryType.FOCUS: self._focus_trajectory,
            TrajectoryType.RELAX: self._relax_trajectory,
            TrajectoryType.MOOD_LIFT: self._mood_lift_trajectory,
            TrajectoryType.STABILIZE: self._stabilize_trajectory
        }
        
    def get_trajectory(self, trajectory_type: TrajectoryType, 
                      duration: float = 300.0, # 5 minutes
                      start_state: Dict = None,
                      target_state: Dict = None) -> Callable[[float], Tuple[float, float]]:
        """
        Get trajectory function for specified type
        Returns: function that takes time and returns (valence, arousal)
        """
        if trajectory_type not in self.trajectories:
            raise ValueError(f"Unknown trajectory type: {trajectory_type}")
            
        return self.trajectories[trajectory_type](duration, start_state, target_state)
    
    def _calm_down_trajectory(self, duration: float, start_state: Dict = None, 
                             target_state: Dict = None) -> Callable:
        """Exponential decay to low arousal, neutral valence"""
        start_valence = start_state['valence'] if start_state else 0.0
        start_arousal = start_state['arousal'] if start_state else 0.8
        
        target_valence = target_state['valence'] if target_state else 0.1
        target_arousal = target_state['arousal'] if target_state else -0.3
        
        def trajectory(t: float) -> Tuple[float, float]:
            if t >= duration:
                return target_valence, target_arousal
                
            # Exponential decay with time constant tau
            tau = duration / 3.0
            alpha = np.exp(-t / tau)
            
            valence = target_valence + (start_valence - target_valence) * alpha
            arousal = target_arousal + (start_arousal - target_arousal) * alpha
            
            return float(valence), float(arousal)
            
        return trajectory
    
    def _energize_trajectory(self, duration: float, start_state: Dict = None,
                            target_state: Dict = None) -> Callable:
        """Sigmoid rise to high arousal, positive valence"""
        start_valence = start_state['valence'] if start_state else -0.2
        start_arousal = start_state['arousal'] if start_state else -0.3
        
        target_valence = target_state['valence'] if target_state else 0.6
        target_arousal = target_state['arousal'] if target_state else 0.7
        
        def trajectory(t: float) -> Tuple[float, float]:
            if t >= duration:
                return target_valence, target_arousal
                
            # Sigmoid activation
            x = (t / duration) * 12 - 6  # Map to [-6, 6]
            sigmoid = 1 / (1 + np.exp(-x))
            
            valence = start_valence + (target_valence - start_valence) * sigmoid
            arousal = start_arousal + (target_arousal - start_arousal) * sigmoid
            
            return float(valence), float(arousal)
            
        return trajectory
    
    def _focus_trajectory(self, duration: float, start_state: Dict = None,
                         target_state: Dict = None) -> Callable:
        """Gradual rise to moderate arousal, neutral-positive valence"""
        start_valence = start_state['valence'] if start_state else 0.0
        start_arousal = start_state['arousal'] if start_state else 0.0
        
        target_valence = target_state['valence'] if target_state else 0.3
        target_arousal = target_state['arousal'] if target_state else 0.4
        
        def trajectory(t: float) -> Tuple[float, float]:
            if t >= duration:
                return target_valence, target_arousal
                
            # Linear rise with slight overshoot and settle
            progress = t / duration
            if progress < 0.7:
                # Rising phase
                factor = progress / 0.7
                overshoot = 1.2  # 20% overshoot
            else:
                # Settling phase
                settle_progress = (progress - 0.7) / 0.3
                overshoot = 1.2 - 0.2 * settle_progress
                
            valence = start_valence + (target_valence - start_valence) * progress
            arousal = start_arousal + (target_arousal - start_arousal) * progress * overshoot
            
            return float(valence), float(arousal)
            
        return trajectory
    
    def _relax_trajectory(self, duration: float, start_state: Dict = None,
                         target_state: Dict = None) -> Callable:
        """Slow decay to low arousal, slightly positive valence"""
        start_valence = start_state['valence'] if start_state else 0.0
        start_arousal = start_state['arousal'] if start_state else 0.5
        
        target_valence = target_state['valence'] if target_state else 0.4
        target_arousal = target_state['arousal'] if target_state else -0.4
        
        def trajectory(t: float) -> Tuple[float, float]:
            if t >= duration:
                return target_valence, target_arousal
                
            # Double exponential for smooth relaxation
            tau1 = duration / 4.0  # Fast component
            tau2 = duration / 1.5  # Slow component
            
            alpha1 = np.exp(-t / tau1)
            alpha2 = np.exp(-t / tau2)
            combined_alpha = 0.3 * alpha1 + 0.7 * alpha2
            
            valence = target_valence + (start_valence - target_valence) * combined_alpha * 0.5
            arousal = target_arousal + (start_arousal - target_arousal) * combined_alpha
            
            return float(valence), float(arousal)
            
        return trajectory
    
    def _mood_lift_trajectory(self, duration: float, start_state: Dict = None,
                             target_state: Dict = None) -> Callable:
        """Gradual rise in valence, stable moderate arousal"""
        start_valence = start_state['valence'] if start_state else -0.5
        start_arousal = start_state['arousal'] if start_state else -0.2
        
        target_valence = target_state['valence'] if target_state else 0.7
        target_arousal = target_state['arousal'] if target_state else 0.2
        
        def trajectory(t: float) -> Tuple[float, float]:
            if t >= duration:
                return target_valence, target_arousal
                
            # Square root curve for natural mood lifting
            progress = np.sqrt(t / duration)
            
            valence = start_valence + (target_valence - start_valence) * progress
            arousal = start_arousal + (target_arousal - start_arousal) * progress * 0.7
            
            return float(valence), float(arousal)
            
        return trajectory
    
    def _stabilize_trajectory(self, duration: float, start_state: Dict = None,
                             target_state: Dict = None) -> Callable:
        """Oscillating dampening to neutral state"""
        start_valence = start_state['valence'] if start_state else 0.0
        start_arousal = start_state['arousal'] if start_state else 0.0
        
        target_valence = target_state['valence'] if target_state else 0.0
        target_arousal = target_state['arousal'] if target_state else 0.0
        
        def trajectory(t: float) -> Tuple[float, float]:
            if t >= duration:
                return target_valence, target_arousal
                
            # Damped oscillation
            omega = 2 * np.pi / (duration / 3)  # 3 cycles
            decay = np.exp(-t / (duration / 2))
            
            val_offset = (start_valence - target_valence) * decay * np.cos(omega * t)
            aro_offset = (start_arousal - target_arousal) * decay * np.cos(omega * t + np.pi/4)
            
            valence = target_valence + val_offset
            arousal = target_arousal + aro_offset
            
            return float(valence), float(arousal)
            
        return trajectory
