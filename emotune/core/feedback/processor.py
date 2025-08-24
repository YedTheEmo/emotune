import numpy as np
from typing import Dict, List, Tuple
from emotune.core.feedback.collector import FeedbackCollector, FeedbackEvent, FeedbackType

class FeedbackProcessor:
    def __init__(self, collector: FeedbackCollector):
        self.collector = collector
        self.learning_rate = 0.1
        self.reward_decay = 0.95
        
    def process_feedback(self, session_id: str, feedback_data: Dict) -> Dict:
        """
        Analyzes user feedback and translates it into musical adjustments and an impact statement.
        """
        rating = int(np.clip(feedback_data.get('rating', 3), 1, 5))
        comfort = int(np.clip(feedback_data.get('comfort', 5), 1, 10))
        effectiveness = int(np.clip(feedback_data.get('effectiveness', 5), 1, 10))

        adjustments = {}
        impact_statements = []

        # --- Analyze Effectiveness ---
        if effectiveness < 4:
            adjustments['repetition_factor'] = -0.15  # Reduce repetition
            adjustments['rhythm_complexity'] = 0.1   # Increase complexity
            impact_statements.append("You found the music ineffective, so the system will explore more varied and complex patterns.")
        elif effectiveness > 7:
            adjustments['repetition_factor'] = 0.15  # Increase repetition
            impact_statements.append("You found the music effective, so the system will reinforce the current musical theme.")

        # --- Analyze Comfort ---
        if comfort < 4:
            adjustments['dissonance_level'] = -0.1  # Soften
            adjustments['warmth'] = 0.08            # Increase warmth
            impact_statements.append("You felt uncomfortable, so the music will be made less dissonant and warmer.")
        elif comfort > 7:
            adjustments['brightness'] = 0.05
            impact_statements.append("You felt comfortable, so the texture will be a bit brighter.")

        # --- Overall sentiment (rating 1..5) nudges tempo/brightness ---
        sentiment_nudge = (rating - 3) / 2.0  # -1..+1 scaled
        adjustments['tempo_bpm'] = adjustments.get('tempo_bpm', 0.0) + 2.0 * sentiment_nudge
        adjustments['brightness'] = adjustments.get('brightness', 0.0) + 0.05 * sentiment_nudge

        return {
            'adjustments': adjustments,
            'impact': " ".join(impact_statements) if impact_statements else "Thanks for your feedback.",
            'rating': rating,
            'comfort': comfort,
            'effectiveness': effectiveness
        }
        
    def compute_reward_signal(self, trajectory_deviation: float, 
                            emotion_stability: float) -> float:
        """Compute reward signal for RL training."""
        feedback_score, feedback_confidence = self.collector.get_recent_feedback_score()
        
        # Trajectory adherence reward (lower deviation is better)
        trajectory_reward = 1.0 - trajectory_deviation
        
        # Emotion stability reward (lower variance is good)
        # Normalize covariance trace (emotion_stability) into [0,1] range using a capped max.
        # A reasonable cap prevents extreme values from collapsing the reward.
        STABILITY_CAP = 1.0  # fallback default; will be effectively 1.0 if caller already supplies [0,1]
        try:
            # If passed a large numeric, cap and normalize; if already [0,1], this is a no-op
            norm = float(emotion_stability)
            if not np.isfinite(norm):
                norm = STABILITY_CAP
            # If caller passes covariance trace, estimate a scale (heuristic) to keep in [0,1]
            # Use a soft cap by dividing by a constant and clipping.
            # Choose 1.0 as default so existing callers with normalized values are unchanged.
            uncertainty_norm = np.clip(norm / max(STABILITY_CAP, 1e-6), 0.0, 1.0)
        except Exception:
            uncertainty_norm = 1.0
        stability_reward = 1.0 - uncertainty_norm
        
        # User feedback reward, weighted by confidence
        user_reward = feedback_score * feedback_confidence
        
        # Combine rewards
        total_reward = (0.4 * trajectory_reward) + (0.5 * user_reward) + (0.1 * stability_reward)
        
        return np.clip(total_reward, -1.0, 1.0)
        
