import time
import threading
import numpy as np
from typing import Dict, Optional, Callable
from dataclasses import dataclass
import time
import queue
import uuid

from ..emotion.emotion_state import EmotionState
from ..emotion.kalman_filter import KalmanEmotionFilter
from ..emotion.analyzers import EmotionAnalyzer
from ..emotion.capture import EmotionCapture
from ..emotion.fusion import EmotionFusion
from ..trajectory.planner import TrajectoryPlanner
from ..trajectory.dtw_matcher import DTWMatcher
from ..trajectory.library import TrajectoryType
from ..music.rl_agent import RLAgent
from ..music.renderer import MusicRenderer
from ..music.parameter_space import MusicParameterSpace
from ..music.base_mapping import BaseMappingEngine
from ..feedback.collector import FeedbackCollector
from ..feedback.processor import FeedbackProcessor
from ..music.engine_tidal import TidalEngine
from ..music.engine_sonicpi import SonicPiEngine
from ..music.engine_midi import MidiEngine

from utils.logging import get_logger
logger = get_logger()

@dataclass
class SessionConfig:
    emotion_update_rate: float = 1.0  # Hz
    music_update_rate: float = 0.2  # Hz (every 5 seconds)
    trajectory_name: str = "calm_down"
    session_duration: float = 600.0  # 10 minutes default
    enable_rl: bool = True
    enable_feedback: bool = True
    enable_face_analysis: bool = True
    enable_voice_analysis: bool = True
    device: str = 'cpu'
    camera_index: int = 0
    audio_device_index: Optional[int] = None

class SessionManager:
    def __init__(self, config: SessionConfig, db=None):
        self.config = config
        self.running = False
        self.session_start_time = 0.0
        self.shutdown_event = threading.Event()  # Added for robust shutdown

        # Initialize all subsystems
        self._init_subsystems()

        # Threading
        self.emotion_thread = None
        self.music_thread = None
        self.main_loop_thread = None
        self.capture_thread = None

        # Data queues for thread communication
        self.emotion_data_queue = queue.Queue(maxsize=10)
        self.latest_emotion_data = None
        self.emotion_data_lock = threading.Lock()

        # Callbacks
        self.on_emotion_update: Optional[Callable] = None
        self.on_music_update: Optional[Callable] = None
        self.on_session_end: Optional[Callable] = None

    def _init_subsystems(self):
        """Initialize all EmoTune subsystems"""
        # Core components
        self.emotion_state = EmotionState()
        self.kalman_filter = KalmanEmotionFilter()
        self.trajectory_planner = TrajectoryPlanner()
        self.dtw_matcher = DTWMatcher()

        # Emotion analysis components
        self.emotion_analyzer = EmotionAnalyzer(device=self.config.device)

        self.emotion_capture = EmotionCapture(
            face_fps=int(self.config.emotion_update_rate),
            audio_duration=3.0,
            sample_rate=16000
        )
        self.emotion_fusion = EmotionFusion()

        # Music system
        self.param_space = MusicParameterSpace()
        self.base_mapping = BaseMappingEngine()
        self.rl_agent = RLAgent(self.param_space) if self.config.enable_rl else None
        self.music_renderer = MusicRenderer(self.param_space)
        self.music_engine = self._select_music_engine()

        # Feedback system
        if self.config.enable_feedback:
            self.feedback_collector = FeedbackCollector()
            self.feedback_processor = FeedbackProcessor(self.feedback_collector)

        try:
            traj_type = TrajectoryType[self.config.trajectory_name.upper()]
        except KeyError:
            raise ValueError(f"Unknown trajectory name: {self.config.trajectory_name}")

        # Initialize trajectory
        self.trajectory_planner.start_trajectory(
            trajectory_type=traj_type,
            duration=self.config.session_duration,
            start_state=None,
            target_state=None
        )

    def process_emotion_data(self, session_id: str, data: dict) -> dict:
        """
        Handle real‐time emotion_data events from the client over SocketIO.
        Currently a stub: runs the internal update steps and returns
        a minimal structure for the client.
        """
        import time
        logger.info(f"[SessionManager] process_emotion_data for session {session_id}")

        # Compute elapsed time since session start
        session_time = time.time() - getattr(self, 'session_start_time', time.time())

        # Run the internal update pipelines (feel free to integrate `data` here)
        try:
            self._process_emotion_update(session_time)
            self._process_music_update(session_time)
        except Exception as ex:
            logger.error(f"Error in processing pipeline: {ex}")

        # Fetch the latest status
        status = self.get_current_status()

        return {
            'emotion_state': status.get('emotion_state'),
            'trajectory_progress': status.get('trajectory_progress'),
            'music_parameters': status.get('music_parameters'),
        }

    def start_emotion_monitoring(self, session_id: str):
        """
        Called when the client asks to begin streaming emotion data over WebSocket.
        The actual capture/analysis threads are already running after start_session().
        """
        logger.info(f"[SessionManager] Emotion monitoring activated for session {session_id}")
        # no-op beyond validation, since _emotion_capture_loop is already feeding data

    def start_session(self, trajectory_type='calm_down', duration=300) -> str:
        """Start the EmoTune session and return a unique session ID"""
        if self.running:
            raise RuntimeError("A session is already running.")

        try:
            # Generate and store a unique session ID
            session_id = str(uuid.uuid4())
            self.current_session_id = session_id

            # Initialize emotion capture
            if not self.emotion_capture.start():
                logger.error("Failed to start emotion capture")
                raise RuntimeError("Emotion capture start failed")

            self.shutdown_event.clear()  # Reset shutdown event at session start
            self.running = True
            self.session_start_time = time.time()

            # Store session parameters
            self.current_trajectory_type = trajectory_type
            self.current_duration = duration

            # Reset emotion state and feedback
            self.emotion_state.reset()
            if self.config.enable_feedback:
                self.feedback_collector.reset_session()

            # Start processing threads
            self.capture_thread = threading.Thread(target=self._emotion_capture_loop, name="EmotionCaptureThread")
            self.capture_thread.start()

            self.main_loop_thread = threading.Thread(target=self._main_processing_loop, name="MainProcessingThread")
            self.main_loop_thread.start()

            logger.info(f"Session {session_id} started with trajectory {trajectory_type}, duration {duration}s")
            return session_id

        except Exception as e:
            logger.error(f"Failed to start session: {e}")
            self.running = False
            self.shutdown_event.set()
            raise

    def stop_session(self, session_id: str = None):
        """Stop the EmoTune session if session ID matches (or always if session_id is None)"""
        if not self.running:
            raise RuntimeError("No session is currently running.")

        if session_id is not None and session_id != getattr(self, 'current_session_id', None):
            raise ValueError("Invalid or expired session ID.")

        logger.info(f"Stopping session {getattr(self, 'current_session_id', session_id)}...")
        self.running = False
        self.shutdown_event.set()  # Signal all threads to stop

        # Stop emotion capture
        if hasattr(self, 'emotion_capture'):
            self.emotion_capture.stop()

        # Wait for threads to finish
        for thread_name, thread in [("capture_thread", self.capture_thread), ("main_loop_thread", self.main_loop_thread)]:
            if thread and thread.is_alive():
                thread.join(timeout=2.0)
                if thread.is_alive():
                    logger.warning(f"{thread_name} did not terminate after shutdown signal!")

        # Session cleanup
        if self.on_session_end:
            session_summary = self._generate_session_summary()
            self.on_session_end(session_summary)

        logger.info(f"Session {getattr(self, 'current_session_id', session_id)} stopped")

    def _emotion_capture_loop(self):
        """Capture emotion data from sensors and queue for processing"""
        while not self.shutdown_event.is_set():
            try:
                # Get data from emotion capture (a dict with keys 'face_frame' & 'audio_chunk')
                data = self.emotion_capture.get_data()
                if not data:
                    self.shutdown_event.wait(0.05)
                    continue

                # Explicitly extract the numpy arrays (or None) from the dict
                face_frame   = data.get('face_frame')
                audio_chunk  = data.get('audio_chunk')     

                if face_frame is not None or audio_chunk is not None:
                    # Analyze emotions
                    analysis_result = self.emotion_analyzer.analyze(
                        face_frame=face_frame,
                        audio_chunk=audio_chunk
                    )
                    
                    # Fuse the emotion data
                    fused_emotion = self._fuse_emotion_data(analysis_result)
                    
                    if fused_emotion is not None:
                        with self.emotion_data_lock:
                            self.latest_emotion_data = fused_emotion
                        try:
                            self.emotion_data_queue.put_nowait(fused_emotion)
                        except queue.Full:
                            try:
                                self.emotion_data_queue.get_nowait()
                                self.emotion_data_queue.put_nowait(fused_emotion)
                            except queue.Empty:
                                pass

                self.shutdown_event.wait(0.05)  # Use event wait for interruptible sleep
            except Exception as e:
                logger.error(f"Error in emotion capture loop: {e}")
                self.shutdown_event.wait(0.1)

    def _fuse_emotion_data(self, analysis_result: Dict) -> Optional[Dict]:
        try:
            face_data = analysis_result.get('face')
            voice_data = analysis_result.get('voice')

            # Build face dict if valid
            face_dict = None
            if face_data and face_data.get('confidence', 0) > 0.1:
                face_dict = {
                    'emotions': {
                        'valence': face_data['emotions']['valence'],
                        'arousal': face_data['emotions']['arousal']
                    },
                    'confidence': face_data['confidence']
                }

            # Build voice dict if valid
            voice_dict = None
            if voice_data and voice_data.get('confidence', 0) > 0.1:
                voice_dict = {
                    'emotions': {
                        'valence': voice_data['emotions']['valence'],
                        'arousal': voice_data['emotions']['arousal']
                    },
                    'confidence': voice_data['confidence']
                }

            # Call fuse_emotions with the correct dict parameters
            fused_emotion = self.emotion_fusion.fuse_emotions(
                face_data=face_dict,
                voice_data=voice_dict
            )

            return fused_emotion

        except Exception as e:
            logger.error(f"Error fusing emotion data: {e}")
            return None

    def _main_processing_loop(self):
        """Main processing loop that coordinates all subsystems"""
        last_emotion_update = 0.0
        last_music_update = 0.0

        emotion_interval = 1.0 / self.config.emotion_update_rate
        music_interval = 1.0 / self.config.music_update_rate

        while not self.shutdown_event.is_set():
            current_time = time.time()
            session_time = current_time - self.session_start_time

            # Check if session should end
            if session_time >= self.config.session_duration:
                self.stop_session()
                break

            # Emotion processing
            if current_time - last_emotion_update >= emotion_interval:
                self._process_emotion_update(session_time)
                last_emotion_update = current_time

            # Music processing
            if current_time - last_music_update >= music_interval:
                self._process_music_update(session_time)
                last_music_update = current_time

            self.shutdown_event.wait(0.05)

    def _process_emotion_update(self, session_time: float):
        """Process emotion sensing and filtering"""
        try:
            fused_emotion = self._get_current_emotion()  # now returns full dict with mean + covariance

            if fused_emotion is not None:
                # Kalman expects valence/arousal, not mean/covariance
                filtered_state, _ = self.kalman_filter.update({
                    'valence': fused_emotion['valence'],
                    'arousal': fused_emotion['arousal'],
                    'uncertainty': fused_emotion.get('uncertainty', 0.5)
                })

                filtered_emotion = {
                    'valence': filtered_state[0],
                    'arousal': filtered_state[1]
                }

                logger.info(f"Updating emotion with filtered data: {filtered_emotion}")

                # Now also store full uncertainty trace
                self.emotion_state.update_emotion({
                    'mean': filtered_emotion,
                    'covariance': fused_emotion['covariance'],
                    'uncertainty_trace': fused_emotion.get('uncertainty')
                })

                target_emotion = self.trajectory_planner.get_current_target()
                deviation = self.dtw_matcher.compute_trajectory_deviation(
                    self.emotion_state.get_emotion_trajectory(),
                    self.trajectory_planner.current_trajectory,
                    self.trajectory_planner.start_time
                )

                if self.rl_agent:
                    self._store_rl_state(filtered_emotion, deviation, session_time)

                if self.on_emotion_update:
                    self.on_emotion_update({
                        'emotion': filtered_emotion,
                        'target': target_emotion,
                        'deviation': deviation,
                        'session_time': session_time
                    })

        except Exception as e:
            logger.error(f"Error in emotion processing: {e}")


    def _get_current_emotion(self) -> Optional[np.ndarray]:
        """Get current emotion from sensors"""
        try:
            # Try to get emotion from queue first (most recent)
            try:
                emotion_data = self.emotion_data_queue.get_nowait()
                return emotion_data
            except queue.Empty:
                pass

            # Fall back to latest emotion data
            with self.emotion_data_lock:
                if self.latest_emotion_data is not None:
                    return self.latest_emotion_data.copy()

            # No emotion data available
            return None

        except Exception as e:
            logger.error(f"Error getting current emotion: {e}")
            return None

    def _get_rl_state(self, session_time: float) -> np.ndarray:
        """Get current state vector for RL agent"""
        current_emotion = self.emotion_state.get_current_emotion()
        emotion_cov = current_emotion.get('covariance') if current_emotion else None

        # Get trajectory progress
        progress = session_time / self.config.session_duration

        # Get DTW deviation
        target = self.trajectory_planner.get_current_target()
        recent_trajectory = self.emotion_state.get_emotion_trajectory()
        deviation = self.dtw_matcher.compute_trajectory_deviation(
            recent_trajectory,
            self.trajectory_planner.current_trajectory,
            self.trajectory_planner.start_time
        )

        if current_emotion:
            emotion_mean = np.array([
                current_emotion['mean']['valence'],
                current_emotion['mean']['arousal']
            ])
        else:
            # fallback zero vector if no emotion
            emotion_mean = np.array([0.0, 0.0])

        return self.rl_agent.get_state_vector(
            emotion_mean, emotion_cov, deviation, progress
        )

    def _store_rl_state(self, emotion: np.ndarray, deviation: float, session_time: float):
        """Store state for RL training"""
        if not self.rl_agent:
            return

        try:
            # Create state-action-reward tuple for RL training
            current_state = self._get_rl_state(session_time)
            
            # Store the experience in RL agent's memory
            # This would typically include: state, action, reward, next_state, done
            self.rl_agent.store_experience(
                state=current_state,
                emotion=emotion,
                deviation=deviation,
                session_time=session_time
            )
            
        except Exception as e:
            logger.error(f"Error storing RL state: {e}")

    def _process_rl_training(self, session_time: float):
        """Process RL training updates using explicit/implicit feedback for reward."""
        if not self.config.enable_feedback or not self.rl_agent:
            return

        try:
            # Get reward signal using only explicit/implicit feedback
            trajectory_deviation = self.dtw_matcher.compute_trajectory_deviation(
                self.emotion_state.get_emotion_trajectory(),
                self.trajectory_planner.current_trajectory,
                self.trajectory_planner.start_time
            )
            emotion_stability = self.emotion_state.get_stability_metric()
            reward = self.feedback_processor.compute_reward_signal(
                trajectory_deviation, emotion_stability
            )

            # Update RL agent with reward
            self.rl_agent.update_reward(reward)

            # Train RL agent periodically
            if int(session_time) % 30 == 0:  # Every 30 seconds
                logger.info("Training RL agent...")
                self.rl_agent.train()

        except Exception as e:
            logger.error(f"Error in RL training: {e}")

    def _generate_session_summary(self) -> Dict:
        """Generate summary of completed session"""
        session_duration = time.time() - self.session_start_time

        summary = {
            'session_duration': session_duration,
            'trajectory_name': self.config.trajectory_name,
            'emotion_updates': self.emotion_state.get_update_count(),
            'final_emotion': self.emotion_state.get_current_emotion(),
            'trajectory_adherence': self.trajectory_planner.evaluate_trajectory_adherence(),
            'emotion_analysis_enabled': {
                'face': self.config.enable_face_analysis,
                'voice': self.config.enable_voice_analysis
            }
        }

        if self.config.enable_feedback:
            feedback_summary = self.feedback_collector.get_session_feedback_summary()
            summary['feedback'] = feedback_summary

        if self.rl_agent:
            rl_summary = self.rl_agent.get_training_summary()
            summary['rl_training'] = rl_summary

        return summary

    # Public interface methods
    def submit_feedback(self, rating: float, context: str = ""):
        """Submit explicit user feedback"""
        if self.config.enable_feedback:
            self.feedback_collector.collect_explicit_feedback(rating, context)

    def log_interaction(self, interaction_type: str, intensity: float = 1.0):
        """Log user interaction for implicit feedback"""
        if self.config.enable_feedback:
            self.feedback_collector.collect_implicit_feedback(interaction_type, intensity)

 
 
    def get_emotion_analysis_status(self) -> Dict:
        """Get detailed emotion analysis status"""
        return {
            'analyzers_loaded': {
                'face': hasattr(self.emotion_analyzer, 'face_analyzer'),
                'voice': hasattr(self.emotion_analyzer, 'voice_analyzer')
            },
            'capture_config': {
                'face_enabled': self.config.enable_face_analysis,
                'voice_enabled': self.config.enable_voice_analysis,
                'camera_index': self.config.camera_index,
                'audio_device_index': self.config.audio_device_index
            },
            'processing_stats': {
                'emotion_queue_size': self.emotion_data_queue.qsize(),
                'has_latest_data': self.latest_emotion_data is not None
            }
        }

    def get_current_status(self) -> Dict:
        """Get current session status"""
        session_time = time.time() - self.session_start_time if self.running else 0

        # Get latest emotion data safely
        with self.emotion_data_lock:
            latest_raw_emotion = self.latest_emotion_data

        # Defensive logging
        if latest_raw_emotion is not None:
            logger.info(f"[get_current_status] latest_raw_emotion type: {type(latest_raw_emotion)}")

        def safe_tolist(obj):
            try:
                return obj.tolist()
            except AttributeError:
                return obj

        return {
            'running': self.running,
            'session_time': session_time,
            'progress': session_time / self.config.session_duration if self.running else 0,
            'current_emotion': self.emotion_state.get_current_emotion(),
            'latest_raw_emotion': safe_tolist(latest_raw_emotion),
            'target_emotion': self.trajectory_planner.get_current_target() if self.running else None,
            'music_parameters': self.music_renderer.current_params,
            'emotion_capture_running': self.emotion_capture.is_running() if hasattr(self.emotion_capture, 'is_running') else False,
            'queue_size': self.emotion_data_queue.qsize()
        }

    def update_emotion_config(self, **kwargs):
        """Update emotion analysis configuration during runtime"""
        config_updated = False
        
        if 'enable_face_analysis' in kwargs:
            self.config.enable_face_analysis = kwargs['enable_face_analysis']
            config_updated = True
            
        if 'enable_voice_analysis' in kwargs:
            self.config.enable_voice_analysis = kwargs['enable_voice_analysis']
            config_updated = True
            
        if config_updated and hasattr(self, 'emotion_capture'):
            # Restart emotion capture with new config
            self.emotion_capture.stop()
            self.emotion_capture = EmotionCapture(
                camera_index=self.config.camera_index,
                audio_device_index=self.config.audio_device_index,
                enable_face=self.config.enable_face_analysis,
                enable_voice=self.config.enable_voice_analysis
            )
            if self.running:
                self.emotion_capture.start()
                
        logger.info(f"Updated emotion config: {kwargs}")

    def _select_music_engine(self):
        """Select and instantiate the music engine based on config or environment."""
        # TODO: Make this configurable (e.g., from config or env var)
        engine_type = getattr(self.config, 'music_engine', 'tidal').lower()
        if engine_type == 'tidal':
            return TidalEngine()
        elif engine_type == 'sonicpi':
            return SonicPiEngine()
        elif engine_type == 'midi':
            return MidiEngine()
        else:
            logger.warning(f"Unknown music engine '{engine_type}', defaulting to TidalEngine.")
            return TidalEngine()

    def _process_music_update(self, session_time: float):
        """Update music parameters and send to engine."""
        try:
            # Interpolate parameters toward target
            params = self.music_renderer.interpolate_parameters()
            # Render music structure
            music_struct = self.music_renderer.render(params)
            # Send to engine
            if self.music_engine:
                self.music_engine.play(music_struct)
            if self.on_music_update:
                self.on_music_update(music_struct)
        except Exception as e:
            logger.error(f"Error in music update: {e}")

    # --- Backend music control stubs ---
    def play_music(self):
        if self.music_engine:
            # Could re-send last rendered music_struct
            params = self.music_renderer.current_params
            music_struct = self.music_renderer.render(params)
            self.music_engine.play(music_struct)

    def pause_music(self):
        if self.music_engine:
            self.music_engine.stop()

    def regenerate_music(self):
        if self.music_engine:
            # Optionally randomize or reset parameters
            params = self.music_renderer.param_space.get_default_parameters()
            self.music_renderer.update_target_parameters(params)
            music_struct = self.music_renderer.render(params)
            self.music_engine.play(music_struct)
