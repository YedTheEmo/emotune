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
        rating = feedback_data.get('rating', 3)  # Neutral default
        comfort = feedback_data.get('comfort', 5)  # Neutral default
        effectiveness = feedback_data.get('effectiveness', 5)  # Neutral default

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
            adjustments['dissonance_level'] = -0.2  # Reduce dissonance
            adjustments['warmth'] = 0.15            # Increase warmth
            impact_statements.append("You felt uncomfortable, so the music will be made less dissonant and warmer.")
        elif comfort > 7:
            adjustments['dissonance_level'] = 0.1  # Allow slightly more dissonance
            impact_statements.append("You felt comfortable, so the system will maintain the current harmonic structure.")

        # --- Analyze Overall Rating ---
        if rating <= 2: # Negative or Very Negative
            adjustments['tempo_bpm'] = -10         # Slow down
            adjustments['brightness'] = -0.1       # Make it darker
            impact_statements.append("Based on your negative rating, the tempo and brightness will be lowered.")
        elif rating >= 4: # Positive or Very Positive
            adjustments['tempo_bpm'] = 10          # Speed up
            adjustments['brightness'] = 0.1        # Make it brighter
            impact_statements.append("Based on your positive rating, the tempo and brightness will be increased.")

        if not impact_statements:
            impact_statement = "Your feedback is neutral. The system will continue with the current musical direction."
        else:
            impact_statement = " ".join(impact_statements)

        return {
            "adjustments": adjustments,
            "impact_statement": impact_statement
        }
        
    def compute_reward_signal(self, trajectory_deviation: float, 
                            emotion_stability: float) -> float:
        """Compute reward signal for RL training."""
        feedback_score, feedback_confidence = self.collector.get_recent_feedback_score()
        
        # Trajectory adherence reward (lower deviation is better)
        trajectory_reward = 1.0 - trajectory_deviation
        
        # Emotion stability reward (lower variance is good)
        stability_reward = 1.0 - emotion_stability
        
        # User feedback reward, weighted by confidence
        user_reward = feedback_score * feedback_confidence
        
        # Combine rewards
        total_reward = (0.4 * trajectory_reward) + (0.5 * user_reward) + (0.1 * stability_reward)
        
        return np.clip(total_reward, -1.0, 1.0)
        
