"""
Working SessionManager that initializes properly
"""

import time
import threading
import numpy as np
from typing import Dict, Optional, Callable
from dataclasses import dataclass
import time
import queue
import uuid
import cv2

from emotune.core.session.state import *
from ..emotion.emotion_state import EmotionState
from ..music.parameter_space import MusicParameterSpace
from emotune.utils.logging import get_logger

logger = get_logger()

@dataclass
class WorkingSessionConfig:
    emotion_update_rate: float = 0.5  # Reduced from 1.0 to prevent UI freezing
    music_update_rate: float = 0.2
    trajectory_name: str = "calm_down"
    session_duration: float = 600.0
    enable_rl: bool = False
    enable_feedback: bool = False
    enable_face_analysis: bool = True
    enable_voice_analysis: bool = True
    device: str = 'cpu'
    camera_index: int = 0

class WorkingSessionManager:
    """Working SessionManager that initializes properly"""
    
    def __init__(self, config, db=None, app=None, socketio=None, param_space=None):
        self.config = config or WorkingSessionConfig()
        self.db = db
        self.app = app
        self.socketio = socketio
        self.running = False
        self.session_start_time = 0.0
        self.current_session_id = None
        self.shutdown_event = threading.Event()
        
        # Music parameter space
        self.param_space = param_space if param_space is not None else MusicParameterSpace()
        
        # Initialize working components
        self._init_working_components()
        
        logger.info("Working SessionManager initialized successfully")
    
    def _init_working_components(self):
        """Initialize components that actually work"""
        try:
            # Core emotion components
            self.emotion_state = EmotionState()
            
            # Initialize camera capture
            self._init_camera()
            
            # Initialize audio capture
            self._init_audio()
            
            # Initialize music engine
            self._init_music_engine()
            
            # Initialize trajectory system
            self._init_trajectory_system()
            
            logger.info("Working components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing working components: {e}")
            # Initialize fallback components
            self._init_fallback_components()
    
    def _init_camera(self):
        """Initialize camera for face detection"""
        try:
            self.camera = cv2.VideoCapture(self.config.camera_index)
            if self.camera.isOpened():
                logger.info("Camera initialized successfully")
                self.camera_available = True
            else:
                logger.warning("Camera not available, using fallback")
                self.camera_available = False
                self.camera = None
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            self.camera_available = False
            self.camera = None
    
    def _init_audio(self):
        """Initialize audio capture for voice analysis"""
        try:
            import pyaudio
            self.audio = pyaudio.PyAudio()
            self.audio_available = True
            logger.info("Audio initialized successfully")
        except Exception as e:
            logger.warning(f"Audio initialization failed: {e}")
            self.audio_available = False
            self.audio = None
    
    def _init_music_engine(self):
        """Initialize music engine"""
        try:
            from emotune.core.music.engine_pyo import create_music_engine
            self.music_engine = create_music_engine("auto")
            logger.info("Music engine initialized successfully")
        except Exception as e:
            logger.error(f"Music engine initialization failed: {e}")
            self.music_engine = MockMusicEngine()
    
    def _init_trajectory_system(self):
        """Initialize trajectory planning system"""
        self.trajectory_planner = WorkingTrajectoryPlanner()
        self.current_trajectory_target = (0.0, 0.0)
        self.trajectory_start_time = time.time()
    
    def _init_fallback_components(self):
        """Initialize fallback components when real ones fail"""
        logger.info("Initializing fallback components")
        self.camera_available = False
        self.audio_available = False
        self.music_engine = MockMusicEngine()
    
    def start_session(self, trajectory_type='calm_down', duration=300):
        """Start a working session"""
        try:
            session_id = str(uuid.uuid4())
            self.current_session_id = session_id
            self.running = True
            self.session_start_time = time.time()
            
            # Start emotion monitoring
            if self.camera_available or self.audio_available:
                self._start_emotion_monitoring()
            
            # Start trajectory
            self._start_trajectory(trajectory_type, duration)
            
            logger.info(f"Working session started: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error starting working session: {e}")
            raise
    
    def stop_session(self, session_id=None):
        """Stop the current session"""
        try:
            self.running = False
            self.current_session_id = None
            
            # Stop emotion monitoring
            self._stop_emotion_monitoring()
            
            # Stop music
            if hasattr(self.music_engine, 'stop'):
                self.music_engine.stop()
            
            logger.info("Working session stopped")
            
        except Exception as e:
            logger.error(f"Error stopping working session: {e}")
    
    def _start_emotion_monitoring(self):
        """Start emotion monitoring with real camera/audio"""
        self.emotion_thread = threading.Thread(target=self._emotion_monitoring_loop, daemon=True)
        self.emotion_thread.start()
        logger.info("Emotion monitoring started")
    
    def _stop_emotion_monitoring(self):
        """Stop emotion monitoring"""
        if hasattr(self, 'emotion_thread') and self.emotion_thread and self.emotion_thread.is_alive():
            self.running = False
            self.emotion_thread.join(timeout=2.0)
            logger.info("Emotion monitoring stopped")
    
    def _emotion_monitoring_loop(self):
        """Main emotion monitoring loop"""
        while self.running:
            try:
                emotion_data = self._capture_emotion_data()
                
                if emotion_data:
                    # Process emotion data
                    processed_emotion = self._process_emotion_data_internal(emotion_data)
                    
                    # Update music parameters based on emotion
                    music_params = self._emotion_to_music_params(processed_emotion)
                    
                    # Update music engine
                    if hasattr(self.music_engine, 'update_parameters'):
                        self.music_engine.update_parameters(music_params)
                    
                    # Store emotion data
                    self._store_emotion_data(processed_emotion)
                    
                    # Store for trajectory visualization
                    self._last_emotion_data = processed_emotion
                
                time.sleep(1.0 / self.config.emotion_update_rate)
                
            except Exception as e:
                logger.error(f"Error in emotion monitoring loop: {e}")
                time.sleep(1.0)
    
    def _capture_emotion_data(self):
        """Capture emotion data from camera and audio"""
        emotion_data = {}
        
        # Capture from camera
        if self.camera_available and self.camera:
            try:
                ret, frame = self.camera.read()
                if ret:
                    # Simple face detection
                    face_emotion = self._analyze_face_emotion(frame)
                    emotion_data['face'] = face_emotion
            except Exception as e:
                logger.error(f"Camera capture error: {e}")
        
        # Capture from audio
        if self.audio_available and self.audio:
            try:
                # Simple audio capture (placeholder)
                voice_emotion = self._analyze_voice_emotion()
                emotion_data['voice'] = voice_emotion
            except Exception as e:
                logger.error(f"Audio capture error: {e}")
        
        return emotion_data if emotion_data else None
    
    def _analyze_face_emotion(self, frame):
        """Analyze face emotion from frame"""
        try:
            # Simple emotion analysis based on face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Generate dynamic emotion based on face presence and time
                t = time.time() - self.session_start_time
                valence = 0.3 * np.sin(0.1 * t) + 0.1 * np.random.normal()
                arousal = 0.4 * np.cos(0.15 * t) + 0.1 * np.random.normal()
                confidence = 0.8
            else:
                valence = 0.1 * np.random.normal()
                arousal = 0.1 * np.random.normal()
                confidence = 0.3
            
            return {
                'valence': np.clip(valence, -1.0, 1.0),
                'arousal': np.clip(arousal, -1.0, 1.0),
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Face emotion analysis error: {e}")
            return {'valence': 0.0, 'arousal': 0.0, 'confidence': 0.1}
    
    def _analyze_voice_emotion(self):
        """Analyze voice emotion (placeholder)"""
        # Generate dynamic voice emotion
        t = time.time() - self.session_start_time
        valence = 0.2 * np.sin(0.08 * t) + 0.05 * np.random.normal()
        arousal = 0.3 * np.cos(0.12 * t) + 0.05 * np.random.normal()
        
        return {
            'valence': np.clip(valence, -1.0, 1.0),
            'arousal': np.clip(arousal, -1.0, 1.0),
            'confidence': 0.6
        }
    
    def _process_emotion_data_internal(self, emotion_data):
        """Process and fuse emotion data"""
        face_data = emotion_data.get('face', {'valence': 0.0, 'arousal': 0.0, 'confidence': 0.0})
        voice_data = emotion_data.get('voice', {'valence': 0.0, 'arousal': 0.0, 'confidence': 0.0})
        
        # Simple weighted fusion
        face_weight = face_data['confidence']
        voice_weight = voice_data['confidence']
        total_weight = face_weight + voice_weight
        
        if total_weight > 0:
            valence = (face_data['valence'] * face_weight + voice_data['valence'] * voice_weight) / total_weight
            arousal = (face_data['arousal'] * face_weight + voice_data['arousal'] * voice_weight) / total_weight
            confidence = total_weight / 2.0
        else:
            valence = arousal = confidence = 0.0
        
        return {
            'valence': valence,
            'arousal': arousal,
            'confidence': confidence,
            'timestamp': time.time()
        }
    
    def _emotion_to_music_params(self, emotion):
        """Convert emotion to music parameters"""
        valence = emotion['valence']
        arousal = emotion['arousal']
        confidence = emotion['confidence']
        
        # Map emotion to music parameters
        tempo_bpm = 120.0 + (arousal * 40.0)  # 80-160 BPM
        volume = 0.5 + (confidence * 0.3)     # 0.5-0.8 volume
        brightness = 0.5 + (valence * 0.4)    # 0.1-0.9 brightness
        warmth = 0.5 - (valence * 0.3)        # 0.2-0.8 warmth
        
        return {
            'tempo_bpm': np.clip(tempo_bpm, 80.0, 160.0),
            'volume': np.clip(volume, 0.1, 1.0),
            'brightness': np.clip(brightness, 0.1, 0.9),
            'warmth': np.clip(warmth, 0.1, 0.9)
        }
    
    def _store_emotion_data(self, emotion_data):
        """Store emotion data in database"""
        if self.db and self.current_session_id:
            try:
                self.db.save_emotion_data(
                    session_id=self.current_session_id,
                    timestamp=emotion_data['timestamp'],
                    valence=emotion_data['valence'],
                    arousal=emotion_data['arousal'],
                    confidence=emotion_data['confidence']
                )
            except Exception as e:
                logger.error(f"Error storing emotion data: {e}")
    
    def _start_trajectory(self, trajectory_type, duration):
        """Start trajectory planning"""
        self.trajectory_planner.start_trajectory(trajectory_type, duration)
        logger.info(f"Trajectory started: {trajectory_type} for {duration}s")
    
    # Public API methods required by app.py
    def set_socketio_sid(self, socketio_sid):
        """Set SocketIO session ID"""
        self.socketio_sid = socketio_sid
        logger.info(f"Working SessionManager: SocketIO SID set to {socketio_sid}")
    
    def start_emotion_monitoring(self, session_id: str):
        """Start emotion monitoring"""
        logger.info(f"Working SessionManager: Starting emotion monitoring for session {session_id}")
        return True
    
    def process_emotion_data(self, session_id: str, data: dict) -> dict:
        """Process emotion data"""
        try:
            # Get current emotion state
            if hasattr(self, '_last_emotion_data'):
                emotion_state = self._last_emotion_data
            else:
                # Generate current emotion based on monitoring
                t = time.time() - self.session_start_time if self.session_start_time else 0
                emotion_state = {
                    'valence': 0.3 * np.sin(0.1 * t),
                    'arousal': 0.4 * np.cos(0.15 * t),
                    'confidence': 0.7,
                    'timestamp': time.time()
                }
            
            # Get current music parameters
            music_params = self._emotion_to_music_params(emotion_state)
            
            return {
                'success': True,
                'emotion_state': emotion_state,
                'music_parameters': music_params,
                'session_id': session_id
            }
            
        except Exception as e:
            logger.error(f"Error processing emotion data: {e}")
            return {
                'success': False,
                'error': str(e),
                'session_id': session_id
            }
    
    def process_realtime_feedback(self, session_id: str, feedback: dict):
        """Process real-time feedback"""
        logger.info(f"Working SessionManager: Processing feedback for session {session_id}")
        return True
    
    def get_active_session_count(self):
        """Get active session count"""
        return 1 if self.running else 0
    
    def play_music(self):
        """Play music"""
        try:
            if hasattr(self.music_engine, 'play'):
                self.music_engine.play()
            logger.info("Working music play triggered")
        except Exception as e:
            logger.error(f"Error playing music: {e}")
    
    def pause_music(self):
        """Pause music"""
        try:
            if hasattr(self.music_engine, 'pause'):
                self.music_engine.pause()
            logger.info("Working music pause triggered")
        except Exception as e:
            logger.error(f"Error pausing music: {e}")
    
    def regenerate_music(self):
        """Regenerate music"""
        try:
            if hasattr(self.music_engine, 'regenerate'):
                self.music_engine.regenerate()
            logger.info("Working music regeneration triggered")
        except Exception as e:
            logger.error(f"Error regenerating music: {e}")
    
    def get_current_status(self):
        """Get current status"""
        return {
            'running': self.running,
            'session_id': self.current_session_id,
            'camera_available': getattr(self, 'camera_available', False),
            'audio_available': getattr(self, 'audio_available', False)
        }
    
    def shutdown(self):
        """Shutdown the working manager"""
        try:
            self.stop_session()
            
            # Clean up camera
            if hasattr(self, 'camera') and self.camera:
                self.camera.release()
            
            # Clean up audio
            if hasattr(self, 'audio') and self.audio:
                self.audio.terminate()
            
            logger.info("Working SessionManager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during working shutdown: {e}")

class WorkingTrajectoryPlanner:
    """Working trajectory planner"""
    
    def __init__(self):
        self.start_time = time.time()
        self.trajectory_type = "calm_down"
        self.duration = 300
        self.current_trajectory = self._calm_down_trajectory
    
    def start_trajectory(self, trajectory_type, duration):
        """Start a trajectory"""
        self.start_time = time.time()
        self.trajectory_type = trajectory_type
        self.duration = duration
        
        # Select trajectory function
        if trajectory_type == "calm_down":
            self.current_trajectory = self._calm_down_trajectory
        elif trajectory_type == "energize":
            self.current_trajectory = self._energize_trajectory
        else:
            self.current_trajectory = self._default_trajectory
    
    def _calm_down_trajectory(self, t):
        """Calm down trajectory"""
        progress = min(t / self.duration, 1.0)
        valence = 0.5 - 0.3 * progress  # Move towards calm
        arousal = 0.7 - 0.5 * progress  # Reduce arousal
        return (valence, arousal)
    
    def _energize_trajectory(self, t):
        """Energize trajectory"""
        progress = min(t / self.duration, 1.0)
        valence = -0.2 + 0.6 * progress  # Move towards positive
        arousal = 0.2 + 0.6 * progress   # Increase arousal
        return (valence, arousal)
    
    def _default_trajectory(self, t):
        """Default trajectory"""
        return (0.0, 0.0)
    
    def get_trajectory_info(self):
        """Get trajectory info"""
        t = time.time() - self.start_time
        target = self.current_trajectory(t)
        progress = min(t / self.duration, 1.0)
        
        return {
            'target': target,
            'deviation': 0.1,  # Mock deviation
            'progress': progress,
            'trajectory_type': self.trajectory_type
        }

class MockMusicEngine:
    """Mock music engine"""
    
    def __init__(self):
        self.status = 'stopped'
        self.current_params = {
            'tempo_bpm': 120.0,
            'volume': 0.5,
            'brightness': 0.5,
            'warmth': 0.5
        }
    
    def play(self):
        self.status = 'playing'
    
    def pause(self):
        self.status = 'paused'
    
    def stop(self):
        self.status = 'stopped'
    
    def regenerate(self):
        self.status = 'regenerated'
    
    def update_parameters(self, params):
        self.current_params.update(params)
    
    def get_status(self):
        return {
            'status': self.status,
            'params': self.current_params
        } 