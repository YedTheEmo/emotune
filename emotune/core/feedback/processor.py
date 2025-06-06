import numpy as np
from typing import Dict, List, Tuple
from emotune.core.feedback.collector import FeedbackCollector, FeedbackEvent, FeedbackType

class FeedbackProcessor:
    def __init__(self, collector: FeedbackCollector):
        self.collector = collector
        self.learning_rate = 0.1
        self.reward_decay = 0.95
        
    def compute_reward_signal(self, trajectory_deviation: float, 
                            emotion_stability: float) -> float:
        """Compute reward signal for RL training using only explicit and implicit feedback."""
        feedback_score, feedback_confidence = self.collector.get_recent_feedback_score()
        
        # Trajectory adherence reward (negative deviation is good)
        trajectory_reward = -trajectory_deviation
        
        # Emotion stability reward (less variance is good for therapeutic goals)
        stability_reward = -emotion_stability
        
        # User feedback reward
        user_reward = feedback_score * feedback_confidence
        
        # Combine rewards with weights from the framework
        alpha, beta, gamma = 0.4, 0.5, 0.1
        total_reward = alpha * trajectory_reward + beta * user_reward + gamma * stability_reward
        
        return np.clip(total_reward, -2.0, 2.0)
    
    def process_feedback_for_learning(self) -> Dict[str, float]:
        """Process feedback to extract learning signals (explicit/implicit only)."""
        recent_feedback = self.collector.current_session_feedback[-10:]  # Last 10 feedback events
        
        if not recent_feedback:
            return {"reward": 0.0, "confidence": 0.0, "trend": 0.0}
        
        # Only use explicit and implicit feedback
        filtered_feedback = [fb for fb in recent_feedback if fb.feedback_type in (FeedbackType.EXPLICIT_RATING, FeedbackType.IMPLICIT_INTERACTION)]
        
        # Calculate trend in feedback
        trend = 0.0
        if len(filtered_feedback) >= 2:
            values = [fb.value for fb in filtered_feedback]
            trend = (values[-1] - values[0]) / len(values)
        
        # Overall reward signal
        total_reward = sum(fb.value * self.feedback_weights.get(fb.feedback_type, 0.0) for fb in filtered_feedback)
        avg_reward = total_reward / len(filtered_feedback) if filtered_feedback else 0.0
        
        # Confidence based on feedback consistency
        values = [fb.value for fb in filtered_feedback]
        confidence = 1.0 / (1.0 + np.std(values)) if len(values) > 1 else 0.5
        
        return {
            "reward": avg_reward,
            "confidence": confidence,
            "trend": trend,
            "feedback_count": len(filtered_feedback)
        }
    
    @property
    def feedback_weights(self):
        return self.collector.feedback_weights
