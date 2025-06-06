import numpy as np
from typing import Dict, List, Optional
from .planner import TrajectoryPlanner, TrajectoryStatus, TrajectoryType
import logging
import time

from emotune.utils.logging import get_logger
logger = get_logger()

class AdaptivePlanner(TrajectoryPlanner):
    """Enhanced trajectory planner with adaptive capabilities"""
    
    def __init__(self):
        super().__init__()
        
        # Adaptation parameters
        self.adaptation_history = []
        self.adaptation_cooldown = 30.0  # seconds
        self.last_adaptation_time = 0.0
        
        # Learning parameters
        self.success_threshold = 0.7
        self.trajectory_success_rates = {
            trajectory_type: 0.5 for trajectory_type in TrajectoryType
        }
        
    def adapt_trajectory(self, emotion_trajectory: List[Dict],
                        current_emotion: Dict) -> bool:
        """Adapt current trajectory based on progress and current state"""
        if not self._is_active():
            return False
            
        current_time = time.time()
        if current_time - self.last_adaptation_time < self.adaptation_cooldown:
            return False  # Too soon to adapt again
            
        # Evaluate current trajectory performance
        evaluation = self.evaluate_trajectory_adherence(emotion_trajectory)
        
        if not evaluation['needs_adaptation']:
            return False
            
        logger.info(f"Adapting trajectory due to deviation: {evaluation['deviation']:.3f}")
        
        # Store adaptation attempt
        self.adaptation_history.append({
            'timestamp': current_time,
            'original_type': self.current_type,
            'deviation': evaluation['deviation'],
            'progress': evaluation['progress']
        })
        
        # Determine adaptation strategy
        adaptation_success = self._execute_adaptation(current_emotion, evaluation)
        
        if adaptation_success:
            self.status = TrajectoryStatus.ADAPTING
            self.last_adaptation_time = current_time
            
        return adaptation_success
    
    def _execute_adaptation(self, current_emotion: Dict, evaluation: Dict) -> bool:
        """Execute trajectory adaptation based on current state"""
        try:
            current_val = current_emotion['mean']['valence']
            current_aro = current_emotion['mean']['arousal']
            
            # Get remaining time
            elapsed = evaluation['elapsed_time']
            remaining_time = max(60.0, self.duration - elapsed)  # At least 1 minute
            
            # Choose adaptation strategy based on current emotion and trajectory type
            new_type = self._select_adaptive_trajectory(
                current_val, current_aro, self.current_type
            )
            
            if new_type == self.current_type:
                # Adjust current trajectory parameters instead
                return self._adjust_trajectory_parameters(current_emotion)
            
            # Switch to new trajectory type
            start_state = {
                'valence': current_val,
                'arousal': current_aro
            }
            
            # Determine appropriate target based on new trajectory type
            target_state = self._get_adaptive_target(new_type, current_emotion)
            
            # Start new adapted trajectory
            success = self.start_trajectory(
                new_type, remaining_time, start_state, target_state
            )
            
            if success:
                logger.info(f"Adapted from {self.current_type.value} to {new_type.value}")
                
            return success
            
        except Exception as e:
            logger.error(f"Adaptation failed: {e}")
            return False
    
    def _select_adaptive_trajectory(self, valence: float, arousal: float,
                                  current_type: TrajectoryType) -> TrajectoryType:
        """Select best trajectory type based on current emotion state"""
        
        # High arousal, negative valence -> calm down
        if arousal > 0.3 and valence < -0.2:
            return TrajectoryType.CALM_DOWN
            
        # Low valence -> mood lift
        elif valence < -0.4:
            return TrajectoryType.MOOD_LIFT
            
        # Low arousal, need energy
        elif arousal < -0.3:
            return TrajectoryType.ENERGIZE
            
        # Moderate state -> focus or stabilize
        elif abs(valence) < 0.3 and abs(arousal) < 0.3:
            # Choose based on success rates
            focus_rate = self.trajectory_success_rates[TrajectoryType.FOCUS]
            stabilize_rate = self.trajectory_success_rates[TrajectoryType.STABILIZE]
            return TrajectoryType.FOCUS if focus_rate > stabilize_rate else TrajectoryType.STABILIZE
            
        # High arousal but positive -> relax
        elif arousal > 0.4 and valence > 0.0:
            return TrajectoryType.RELAX
            
        # Default: stick with current or stabilize
        else:
            return current_type if current_type else TrajectoryType.STABILIZE
    
    def _get_adaptive_target(self, trajectory_type: TrajectoryType,
                           current_emotion: Dict) -> Dict:
        """Get adaptive target state for trajectory type"""
        
        # Default targets based on trajectory type
        target_map = {
            TrajectoryType.CALM_DOWN: {'valence': 0.1, 'arousal': -0.3},
            TrajectoryType.ENERGIZE: {'valence': 0.6, 'arousal': 0.7},
            TrajectoryType.FOCUS: {'valence': 0.3, 'arousal': 0.4},
            TrajectoryType.RELAX: {'valence': 0.4, 'arousal': -0.4},
            TrajectoryType.MOOD_LIFT: {'valence': 0.7, 'arousal': 0.2},
            TrajectoryType.STABILIZE: {'valence': 0.0, 'arousal': 0.0}
        }
        
        base_target = target_map.get(trajectory_type, {'valence': 0.0, 'arousal': 0.0})
        
        # Adjust target based on current state for smoother transitions
        current_val = current_emotion['mean']['valence']
        current_aro = current_emotion['mean']['arousal']
        
        # Moderate the target to avoid extreme jumps
        adjustment_factor = 0.3
        adjusted_target = {
            'valence': base_target['valence'] * (1 - adjustment_factor) + current_val * adjustment_factor,
            'arousal': base_target['arousal'] * (1 - adjustment_factor) + current_aro * adjustment_factor
        }
        
        return adjusted_target
    
    def _adjust_trajectory_parameters(self, current_emotion: Dict) -> bool:
        """Adjust current trajectory parameters without changing type"""
        try:
            # Move the target closer to the user's current state for easier adaptation
            if hasattr(self, 'target_state') and self.target_state is not None:
                current_val = current_emotion['mean']['valence']
                current_aro = current_emotion['mean']['arousal']
                old_target_val = self.target_state['valence']
                old_target_aro = self.target_state['arousal']
                # Weighted average: 60% old target, 40% current state
                new_target = {
                    'valence': old_target_val * 0.6 + current_val * 0.4,
                    'arousal': old_target_aro * 0.6 + current_aro * 0.4
                }
                self.target_state = new_target
                # Update the trajectory function to use the new target
                self.current_trajectory = self.library.get_trajectory(
                    self.current_type, self.duration, self.get_current_state(), self.target_state
                )
                logger.info(f"Adjusted trajectory target to {new_target}")
            # Optionally extend duration if behind schedule
            time_extension = min(120.0, self.duration * 0.3)  # Max 30% extension
            self.duration += time_extension
            logger.info(f"Extended trajectory duration by {time_extension}s")
            return True
        except Exception as e:
            logger.error(f"Parameter adjustment failed: {e}")
            return False

    def get_current_state(self) -> dict:
        """Return the current state (valence/arousal) at the current time."""
        if self.current_trajectory and self.start_time is not None:
            elapsed = max(0.0, time.time() - self.start_time)
            val, aro = self.current_trajectory(elapsed)
            return {'valence': val, 'arousal': aro}
        return {'valence': 0.0, 'arousal': 0.0}

    def get_rl_state_vector(self, current_emotion: dict, dtw_deviation: float) -> list:
        """Return RL state vector: [valence, arousal, confidence, target_val, target_aro, dtw_deviation]"""
        mu = current_emotion['mean']
        sigma = current_emotion.get('covariance', 0.0)
        target = self.get_current_target()
        if target is None:
            target = (0.0, 0.0)
        return [
            mu['valence'],
            mu['arousal'],
            sigma if isinstance(sigma, float) else float(np.mean(sigma)),
            target[0],
            target[1],
            dtw_deviation
        ]

    def apply_rl_action(self, delta_p: dict):
        """Apply RL action (delta_p) to nudge the target state."""
        if hasattr(self, 'target_state') and self.target_state is not None:
            self.target_state['valence'] += delta_p.get('valence', 0.0)
            self.target_state['arousal'] += delta_p.get('arousal', 0.0)
            # Update trajectory function
            self.current_trajectory = self.library.get_trajectory(
                self.current_type, self.duration, self.get_current_state(), self.target_state
            )
            logger.info(f"RL action applied: nudged target to {self.target_state}")

    def compute_rl_reward(self, dtw_deviation: float, feedback: float, delta_p: dict, alpha=1.0, beta=0.5, gamma=0.1) -> float:
        """Compute RL reward: r_t = -α δ_t + β fb,t − γ∥Δp∥²"""
        delta_p_norm = np.sqrt(sum((v ** 2 for v in delta_p.values())))
        reward = -alpha * dtw_deviation + beta * feedback - gamma * (delta_p_norm ** 2)
        return reward
    
    def update_trajectory_success_rate(self, trajectory_type: TrajectoryType,
                                     success: bool):
        """Update success rate for a trajectory type based on outcome"""
        current_rate = self.trajectory_success_rates[trajectory_type]
        
        # Exponential moving average update
        alpha = 0.1
        new_rate = current_rate * (1 - alpha) + (1.0 if success else 0.0) * alpha
        
        self.trajectory_success_rates[trajectory_type] = new_rate
        
        logger.info(f"Updated {trajectory_type.value} success rate: {new_rate:.3f}")
    
    def get_adaptation_statistics(self) -> Dict:
        """Get statistics about trajectory adaptations"""
        if not self.adaptation_history:
            return {
                'total_adaptations': 0,
                'adaptation_rate': 0.0,
                'success_rates': dict(self.trajectory_success_rates)
            }
        
        total_adaptations = len(self.adaptation_history)
        recent_adaptations = [
            adapt for adapt in self.adaptation_history
            if time.time() - adapt['timestamp'] < 3600  # Last hour
        ]
        
        return {
            'total_adaptations': total_adaptations,
            'recent_adaptations': len(recent_adaptations),
            'adaptation_rate': len(recent_adaptations) / 60.0,  # per minute
            'success_rates': dict(self.trajectory_success_rates),
            'last_adaptation': self.adaptation_history[-1]['timestamp'] if self.adaptation_history else None
        }
