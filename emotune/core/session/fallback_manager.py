"""
Fallback SessionManager for when main SessionManager fails
"""

import time
import threading
import uuid
from typing import Dict, Optional, Any
from dataclasses import dataclass
from emotune.utils.logging import get_logger

logger = get_logger()

@dataclass
class FallbackSessionConfig:
    emotion_update_rate: float = 1.0
    music_update_rate: float = 0.2
    trajectory_name: str = "calm_down"
    session_duration: float = 600.0
    enable_rl: bool = False
    enable_feedback: bool = False
    enable_face_analysis: bool = False
    enable_voice_analysis: bool = False
    device: str = 'cpu'
    camera_index: int = 0

class FallbackSessionManager:
    """Fallback SessionManager that provides basic functionality when main one fails"""
    
    def __init__(self, config=None, db=None, app=None, socketio=None, param_space=None):
        self.config = config or FallbackSessionConfig()
        self.db = db
        self.app = app
        self.socketio = socketio
        self.param_space = param_space
        self.running = False
        self.session_start_time = 0.0
        self.current_session_id = None
        
        # Mock emotion data for testing
        self._mock_emotion_data = {
            'valence': 0.0,
            'arousal': 0.0,
            'confidence': 0.5
        }
        
        # Mock music engine
        self._mock_music_engine = MockMusicEngine()
        
        # Mock trajectory planner
        self.trajectory_planner = MockTrajectoryPlanner()
        
        # Mock data persistence
        self.data_persistence = MockDataPersistence()
        
        logger.info("Fallback SessionManager initialized")
    
    def start_session(self, trajectory_type='calm_down', duration=300):
        """Start a mock session"""
        try:
            session_id = str(uuid.uuid4())
            self.current_session_id = session_id
            self.running = True
            self.session_start_time = time.time()
            
            logger.info(f"Fallback session started: {session_id}")
            return session_id
        except Exception as e:
            logger.error(f"Error starting fallback session: {e}")
            raise
    
    def stop_session(self, session_id=None):
        """Stop the current session"""
        try:
            self.running = False
            self.current_session_id = None
            logger.info("Fallback session stopped")
        except Exception as e:
            logger.error(f"Error stopping fallback session: {e}")
    
    def play_music(self):
        """Mock music play"""
        try:
            self._mock_music_engine.play()
            logger.info("Fallback music play triggered")
        except Exception as e:
            logger.error(f"Error in fallback music play: {e}")
    
    def pause_music(self):
        """Mock music pause"""
        try:
            self._mock_music_engine.pause()
            logger.info("Fallback music pause triggered")
        except Exception as e:
            logger.error(f"Error in fallback music pause: {e}")
    
    def regenerate_music(self):
        """Mock music regeneration"""
        try:
            self._mock_music_engine.regenerate()
            logger.info("Fallback music regeneration triggered")
        except Exception as e:
            logger.error(f"Error in fallback music regeneration: {e}")
    
    def set_socketio_sid(self, socketio_sid):
        """Set SocketIO session ID (required by app.py)"""
        self.socketio_sid = socketio_sid
        logger.info(f"Fallback SessionManager: SocketIO SID set to {socketio_sid}")
    
    def start_emotion_monitoring(self, session_id: str):
        """Start emotion monitoring (mock implementation)"""
        logger.info(f"Fallback SessionManager: Starting emotion monitoring for session {session_id}")
        return True
    
    def process_emotion_data(self, session_id: str, data: dict) -> dict:
        """Process emotion data (mock implementation)"""
        try:
            logger.info(f"Fallback SessionManager: Processing emotion data for session {session_id}")
            # Return mock emotion data
            mock_emotion = {
                'valence': 0.0,
                'arousal': 0.0,
                'confidence': 0.5,
                'timestamp': time.time()
            }
            return {
                'success': True,
                'emotion_state': mock_emotion,
                'music_parameters': self._mock_music_engine.current_params,
                'session_id': session_id
            }
        except Exception as e:
            logger.error(f"Error in fallback emotion processing: {e}")
            return {
                'success': False,
                'error': str(e),
                'session_id': session_id
            }
    
    def process_realtime_feedback(self, session_id: str, feedback: dict):
        """Process real-time feedback (mock implementation)"""
        logger.info(f"Fallback SessionManager: Processing feedback for session {session_id}")
        return True
    
    def get_active_session_count(self):
        """Get active session count (mock implementation)"""
        return 1 if self.running else 0
    
    def get_current_status(self):
        """Get current status"""
        return {
            'running': self.running,
            'session_id': self.current_session_id,
            'emotion_state': self._mock_emotion_data,
            'music_status': self._mock_music_engine.get_status()
        }
    
    def shutdown(self):
        """Shutdown the fallback manager"""
        try:
            self.stop_session()
            logger.info("Fallback SessionManager shutdown complete")
        except Exception as e:
            logger.error(f"Error during fallback shutdown: {e}")

class MockMusicEngine:
    """Mock music engine for fallback"""
    
    def __init__(self):
        self.status = 'stopped'
        self.current_params = {
            'tempo_bpm': 120.0,
            'volume': 0.5,
            'brightness': 0.5
        }
    
    def play(self):
        self.status = 'playing'
    
    def pause(self):
        self.status = 'paused'
    
    def regenerate(self):
        self.status = 'regenerated'
    
    def get_status(self):
        return {
            'status': self.status,
            'params': self.current_params
        }

class MockTrajectoryPlanner:
    """Mock trajectory planner for fallback"""
    
    def __init__(self):
        self.start_time = time.time()
        self.current_trajectory = lambda t: (0.0, 0.0)
    
    def get_trajectory_info(self):
        return {
            'target': (0.0, 0.0),
            'deviation': 0.0,
            'progress': 0.0
        }

class MockDataPersistence:
    """Mock data persistence for fallback"""
    
    def get_emotion_history(self, session_id):
        return []
    
    def get_settings(self):
        return {}
    
    def get_active_session_count(self):
        return 0 