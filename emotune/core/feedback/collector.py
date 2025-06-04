import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class FeedbackType(Enum):
    EXPLICIT_RATING = "explicit_rating"
    IMPLICIT_INTERACTION = "implicit_interaction"

@dataclass
class FeedbackEvent:
    timestamp: float
    feedback_type: FeedbackType
    value: float
    confidence: float
    metadata: Dict

class FeedbackCollector:
    def __init__(self):
        self.feedback_history: List[FeedbackEvent] = []
        self.current_session_feedback: List[FeedbackEvent] = []
        
        # Implicit feedback tracking
        self.interaction_count = 0
        self.last_interaction_time = time.time()
        self.session_start_time = time.time()
        
        # Feedback weights for different types
        self.feedback_weights = {
            FeedbackType.EXPLICIT_RATING: 1.0,
            FeedbackType.IMPLICIT_INTERACTION: 0.3
        }
    
    def collect_explicit_feedback(self, rating: float, context: str = "") -> None:
        """Collect explicit user rating (0-10 scale)"""
        normalized_rating = self._normalize_rating(rating)
        
        feedback = FeedbackEvent(
            timestamp=time.time(),
            feedback_type=FeedbackType.EXPLICIT_RATING,
            value=normalized_rating,
            confidence=0.9,  # High confidence for explicit feedback
            metadata={
                "raw_rating": rating,
                "context": context,
                "session_time": time.time() - self.session_start_time
            }
        )
        
        self._store_feedback(feedback)
    
    def collect_implicit_feedback(self, interaction_type: str, intensity: float = 1.0) -> None:
        """Collect implicit feedback from user interactions"""
        current_time = time.time()
        time_since_last = current_time - self.last_interaction_time
        
        # Calculate engagement score based on interaction frequency
        engagement_score = self._calculate_engagement_score(time_since_last, interaction_type)
        
        feedback = FeedbackEvent(
            timestamp=current_time,
            feedback_type=FeedbackType.IMPLICIT_INTERACTION,
            value=engagement_score,
            confidence=0.4,  # Lower confidence for implicit feedback
            metadata={
                "interaction_type": interaction_type,
                "intensity": intensity,
                "time_since_last": time_since_last,
                "total_interactions": self.interaction_count
            }
        )
        
        self._store_feedback(feedback)
        self.interaction_count += 1
        self.last_interaction_time = current_time
    
    def get_recent_feedback_score(self, time_window: float = 30.0) -> Tuple[float, float]:
        """Get weighted feedback score for recent time window"""
        current_time = time.time()
        recent_feedback = [
            fb for fb in self.current_session_feedback 
            if current_time - fb.timestamp <= time_window
        ]
        
        if not recent_feedback:
            return 0.0, 0.0  # score, confidence
        
        # Weighted average of feedback
        total_weight = 0.0
        weighted_sum = 0.0
        confidence_sum = 0.0
        
        for feedback in recent_feedback:
            weight = self.feedback_weights.get(feedback.feedback_type, 0.0)
            weighted_sum += feedback.value * weight * feedback.confidence
            total_weight += weight * feedback.confidence
            confidence_sum += feedback.confidence
        
        if total_weight == 0:
            return 0.0, 0.0
        
        average_score = weighted_sum / total_weight
        average_confidence = confidence_sum / len(recent_feedback)
        
        return average_score, average_confidence
    
    def get_session_feedback_summary(self) -> Dict:
        """Get summary of feedback for current session"""
        if not self.current_session_feedback:
            return {"total_feedback": 0, "average_score": 0.0, "feedback_types": {}}
        
        feedback_by_type = {}
        total_score = 0.0
        
        for feedback in self.current_session_feedback:
            fb_type = feedback.feedback_type.value
            if fb_type not in feedback_by_type:
                feedback_by_type[fb_type] = []
            feedback_by_type[fb_type].append(feedback.value)
            total_score += feedback.value
        
        # Calculate averages by type
        type_averages = {fb_type: np.mean(values) for fb_type, values in feedback_by_type.items()}
        
        return {
            "total_feedback": len(self.current_session_feedback),
            "average_score": total_score / len(self.current_session_feedback),
            "feedback_types": type_averages,
            "session_duration": time.time() - self.session_start_time
        }
    
    def _store_feedback(self, feedback: FeedbackEvent) -> None:
        """Store feedback in both session and global history"""
        self.current_session_feedback.append(feedback)
        self.feedback_history.append(feedback)
    
    def _normalize_rating(self, rating: float) -> float:
        """Normalize rating to [-1, 1] range"""
        return (rating - 5.0) / 5.0
    
    def _calculate_engagement_score(self, time_since_last: float, interaction_type: str) -> float:
        """Calculate engagement score from interaction patterns"""
        # High engagement = frequent interactions
        if time_since_last < 5.0:
            base_score = 0.8
        elif time_since_last < 15.0:
            base_score = 0.4
        else:
            base_score = 0.1
        
        # Adjust based on interaction type
        type_multipliers = {
            "button_click": 1.0,
            "slider_adjust": 1.2,
            "feedback_submit": 1.5,
            "page_visit": 0.8
        }
        
        return base_score * type_multipliers.get(interaction_type, 1.0)
    
    def reset_session(self) -> None:
        """Reset session-specific feedback tracking"""
        self.current_session_feedback = []
        self.interaction_count = 0
        self.session_start_time = time.time()
        self.last_interaction_time = time.time()

