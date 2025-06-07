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
from ..emotion.kalman_filter import KalmanEmotionFilter
from ..emotion.fusion import EmotionFusion
from ..emotion.analyzers import EmotionAnalyzer
from ..emotion.capture import EmotionCapture
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

from emotune.utils.logging import get_logger
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
    def __init__(self, config: SessionConfig, db=None, app=None, socketio=None):
        self.config = config
        self.running = False
        self.session_start_time = 0.0
        self.shutdown_event = threading.Event()  # Added for robust shutdown
        self.app = app  # Store Flask app for context in threads
        self.socketio = socketio  # Store SocketIO instance for emits

        # Initialize all subsystems
        self._init_subsystems()

        # Threading
        self.emotion_thread = None
        self.music_thread = None
        self.capture_thread = None

        # Data queues for thread communication
        self.emotion_data_queue = queue.Queue(maxsize=10)
        self.latest_emotion_data = None
        self.emotion_data_lock = threading.Lock()

        # Callbacks
        self.on_emotion_update: Optional[Callable] = None
        self.on_music_update: Optional[Callable] = None
        self.on_session_end: Optional[Callable] = None

        # Camera configuration for EmotionCapture
        self.emotion_capture = EmotionCapture(
            face_fps=int(self.config.emotion_update_rate),
            audio_duration=0.5,
            sample_rate=16000
        )

    def _init_subsystems(self):
        """Initialize all EmoTune subsystems"""
        # Core components
        self.emotion_state = EmotionState()
        self.kalman_filter = KalmanEmotionFilter()
        self.trajectory_planner = TrajectoryPlanner()
        self.dtw_matcher = DTWMatcher()

        # Emotion analysis components
        self.emotion_analyzer = EmotionAnalyzer(device=self.config.device)

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
        logger.debug(f"[SessionManager] process_emotion_data for session {session_id}")

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

        # --- FLATTEN EMOTION STATE FOR FRONTEND ---
        current_emotion = status.get('current_emotion')
        if current_emotion and 'mean' in current_emotion:
            flat_emotion = {
                'valence': current_emotion['mean'].get('valence', 0),
                'arousal': current_emotion['mean'].get('arousal', 0),
                'confidence': 1 - min(1, current_emotion.get('uncertainty_trace', 0.5)/2)
            }
        else:
            flat_emotion = {'valence': 0, 'arousal': 0, 'confidence': 0}

        logger.log_emotion(
            valence=flat_emotion['valence'],
            arousal=flat_emotion['arousal'],
            confidence=flat_emotion['confidence'],
            source='fused',
            raw_data=current_emotion
        )

        # --- BUILD TRAJECTORY PROGRESS FOR FRONTEND ---
        # This logic assumes you have access to trajectory info and actual/target paths
        # If not, you may need to adjust this to match your actual data pipeline
        trajectory_progress = status.get('trajectory_progress')
        if not trajectory_progress:
            # Fallback: build minimal structure
            trajectory_progress = {
                'info': {},
                'actual_path': [],
                'target_path': [],
                'deviation': None
            }

        return {
            'emotion_state': flat_emotion,
            'trajectory_progress': trajectory_progress,
            'music_parameters': status.get('music_parameters', {})
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

            # Initialize emotion capture FIRST
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

            # Start processing threads AFTER capture is running
            self.capture_thread = threading.Thread(target=self._emotion_capture_loop, name="EmotionCaptureThread")
            self.capture_thread.start()

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
        for thread_name, thread in [("capture_thread", self.capture_thread)]:
            if thread and thread.is_alive():
                thread.join(timeout=2.0)
                if thread.is_alive():
                    logger.warning(f"{thread_name} did not terminate after shutdown signal!")

        # Session cleanup
        if self.on_session_end:
            session_summary = self._generate_session_summary()
            self.on_session_end(session_summary)

        logger.info(f"Session {getattr(self, 'current_session_id', session_id)} stopped")

    def set_socketio_sid(self, socketio_sid):
        """Set the current SocketIO sid for this session (for correct emit targeting)."""
        self._current_socketio_sid = socketio_sid

    def _emotion_capture_loop(self):
        """Capture emotion data from sensors and queue for processing"""
        logger.info("[SessionManager] Starting _emotion_capture_loop thread.")
        while not self.shutdown_event.is_set():
            try:
                # Get data from emotion capture (a dict with keys 'face_frame' & 'audio_chunk')
                data = self.emotion_capture.get_data()
                if not data:
                    time.sleep(0.05)  # Reduce CPU usage if no data
                    continue
                logger.debug(f"[SessionManager] emotion_capture.get_data() returned: {type(data)} {list(data.keys()) if data else data}")
                if not data:
                    logger.debug("[SessionManager] No data from emotion_capture.get_data(). Waiting...")
                    self.shutdown_event.wait(0.05)
                    continue

                # Explicitly extract the numpy arrays (or None) from the dict
                face_frame   = data.get('face_frame')
                audio_chunk  = data.get('audio_chunk')     
                logger.info(f"[SessionManager] Got data from capture: face_frame={'yes' if face_frame is not None else 'no'}, audio_chunk={'yes' if audio_chunk is not None else 'no'}.")
                if audio_chunk is not None:
                    logger.info(f"[SessionManager] Audio chunk shape: {audio_chunk.shape}, dtype: {audio_chunk.dtype}")

                # Transform keys to match analyzer's expected parameters
                transformed_data = {}
                if face_frame is not None:
                    transformed_data['frame'] = face_frame  # EmotionAnalyzer expects 'frame'
                if audio_chunk is not None:
                    transformed_data['audio'] = audio_chunk  # EmotionAnalyzer expects 'audio'
                # Optionally pass timestamp if needed (but not to analyze)
                if 'timestamp' in data:
                    transformed_data['timestamp'] = data['timestamp']

                # Only pass allowed keys to analyze()
                allowed = {'frame', 'audio', 'sr'}
                analyze_kwargs = {k: v for k, v in transformed_data.items() if k in allowed}

                if analyze_kwargs:
                    analysis_result = self.emotion_analyzer.analyze(**analyze_kwargs)
                    logger.debug(f"[SessionManager] emotion_analyzer.analyze result: {analysis_result}")
                    fused_emotion = self._fuse_emotion_data(analysis_result)
                    logger.debug(f"[SessionManager] _fuse_emotion_data returned: {fused_emotion}")
                    session_time = time.time() - getattr(self, 'session_start_time', time.time())
                    # --- Call _process_emotion_update to update filtered emotion and history ---
                    self._process_emotion_update(session_time)
                    logger.debug(f"[SessionManager] Called _process_emotion_update at session_time={session_time}")
                    logger.debug(f"[SessionManager] _latest_filtered_emotion: {getattr(self, '_latest_filtered_emotion', None)}")
                    logger.log_emotion(
                        valence=fused_emotion.get('valence', 0) if fused_emotion else 0,
                        arousal=fused_emotion.get('arousal', 0) if fused_emotion else 0,
                        confidence=fused_emotion.get('uncertainty', 0.5) if fused_emotion else 0.5,
                        source='fused',
                        raw_data=fused_emotion
                    )
                    # --- EMIT TO FRONTEND (SOCKETIO) ---
                    try:
                        if self.app is not None and self.socketio is not None:
                            filtered_emotion = getattr(self, '_latest_filtered_emotion', None)
                            sid = getattr(self, '_current_socketio_sid', None)
                            status = self.get_current_status()
                            if not sid:
                                logger.error("[SessionManager] No SocketIO sid set! Cannot emit to frontend.")
                                continue
                            if filtered_emotion is not None:
                                flat_emotion = {
                                    'valence': filtered_emotion.get('valence', 0),
                                    'arousal': filtered_emotion.get('arousal', 0),
                                    'confidence': filtered_emotion.get('confidence', 0.5)
                                }
                                trajectory_progress = status.get('trajectory_progress', {})
                                music_parameters = status.get('music_parameters', {})

                                payload = {
                                    'emotion_state': flat_emotion,
                                    'trajectory_progress': trajectory_progress,
                                    'music_parameters': music_parameters,
                                    'timestamp': time.time()
                                }

                                # Log type and repr for every value in payload
                                for k, v in payload.items():
                                    logger.info(f"[EmitPayload] {k}: type={type(v)}, repr={repr(v)}")
                                    if callable(v):
                                        logger.error(f"[EmitPayload] ABORT: {k} is a function! Skipping emit.")
                                        raise TypeError(f"Emit payload key '{k}' is a function, not serializable.")
                                # --- PATCH: Robustly check payload for unserializable objects/functions ---
                                def is_serializable(obj):
                                    import numpy as np
                                    import types
                                    if isinstance(obj, (int, float, str, bool, type(None))):
                                        return True
                                    if isinstance(obj, (list, tuple)):
                                        return all(is_serializable(x) for x in obj)
                                    if isinstance(obj, dict):
                                        return all(isinstance(k, str) and is_serializable(v) for k, v in obj.items())
                                    if isinstance(obj, np.ndarray):
                                        return True
                                    if isinstance(obj, types.FunctionType):
                                        return False
                                    return False

                                for k, v in payload.items():
                                    if not is_serializable(v):
                                        logger.error(f"[EmitPayload] UNSERIALIZABLE: {k}: type={type(v)}, repr={repr(v)}")
                                        payload[k] = str(v)

                                logger.info(f"[SessionManager] Emitting emotion_update to sid={sid} with payload: {payload}")
                                self.socketio.emit('emotion_update', payload, room=sid)
                            else:
                                logger.warning("[SessionManager] filtered_emotion is None, skipping emit block.")
                    except Exception as e:
                        logger.error(f"[SessionManager] Failed to emit SocketIO events: {e}")
                else:
                    logger.debug("[SessionManager] No valid data to pass to emotion_analyzer.analyze.")

                self.shutdown_event.wait(0.05)  # Use event wait for interruptible sleep
            except Exception as e:
                logger.error(f"Error in emotion capture loop: {e}")
                self.shutdown_event.wait(0.1)

    def _fuse_emotion_data(self, analysis_result: Dict) -> Optional[Dict]:
        logger.debug(f"[SessionManager] _fuse_emotion_data called with: {analysis_result}")
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
                logger.debug(f"[SessionManager] Valid face_data for fusion: {face_dict}")
            else:
                logger.debug("[SessionManager] No valid face_data for fusion.")

            # Build voice dict if valid
            voice_dict = None
            if voice_data and voice_data.get('confidence', 0.1):
                voice_dict = {
                    'emotions': {
                        'valence': voice_data['emotions']['valence'],
                        'arousal': voice_data['emotions']['arousal']
                    },
                    'confidence': voice_data['confidence']
                }
                logger.debug(f"[SessionManager] Valid voice_data for fusion: {voice_dict}")
            else:
                logger.debug("[SessionManager] No valid voice_data for fusion.")

            # Call fuse_emotions with the correct dict parameters
            logger.debug(f"[SessionManager] Calling emotion_fusion.fuse_emotions(face_data={face_dict}, voice_data={voice_dict})")
            fused_emotion = self.emotion_fusion.fuse_emotions(
                face_data=face_dict,
                voice_data=voice_dict
            )
            logger.debug(f"[SessionManager] emotion_fusion.fuse_emotions result: {fused_emotion}")
            return fused_emotion

        except Exception as e:
            logger.error(f"Error fusing emotion data: {e}")
            return None

    def _process_emotion_update(self, session_time: float):
        """Process emotion sensing and filtering"""
        logger.info(f"[SessionManager] _process_emotion_update called at session_time={session_time}")
        try:
            fused_emotion = self._get_current_emotion()  # now returns full dict with mean + covariance
            logger.debug(f"[SessionManager] _get_current_emotion returned: {fused_emotion}")

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
                self._latest_filtered_emotion = filtered_emotion
                logger.info(f"[SessionManager] Updating emotion with filtered data: {filtered_emotion}")

                # Now also store full uncertainty trace
                self.emotion_state.update_emotion({
                    'mean': filtered_emotion,
                    'covariance': fused_emotion['covariance'],
                    'uncertainty_trace': fused_emotion.get('uncertainty')
                })

                # --- NEW: Map filtered emotion to music parameters, log, and update renderer ---
                val = filtered_emotion['valence']
                ar = filtered_emotion['arousal']
                mapped_params = self.base_mapping.map_emotion_to_parameters(val, ar)
                logger.info(f"[SessionManager] Mapped music parameters: {mapped_params}")
                from emotune.utils.logging import log_music
                log_music(mapped_params, trajectory_type=getattr(self, 'current_trajectory_type', None), trajectory_progress=session_time/self.config.session_duration)
                logger.log_music_generation(
                    parameters=mapped_params,
                    trajectory_type=getattr(self, 'current_trajectory_type', None),
                    trajectory_progress=session_time/self.config.session_duration
                )
                self.music_renderer.update_target_parameters(mapped_params)
                self.music_renderer.current_params = mapped_params.copy()  # Immediate update for frontend

                # Store for frontend emission
                self._latest_music_parameters = mapped_params.copy()

                target_emotion = self.trajectory_planner.get_current_target()
                deviation = self.dtw_matcher.compute_trajectory_deviation(
                    self.emotion_state.get_emotion_trajectory(),
                    self.trajectory_planner.current_trajectory,
                    self.trajectory_planner.start_time
                )
                logger.info(f"[SessionManager] Target emotion: {target_emotion}, Deviation: {deviation}")

                if self.rl_agent:
                    logger.debug("[SessionManager] Storing RL state...")
                    self._store_rl_state(filtered_emotion, deviation, session_time)

                if self.on_emotion_update:
                    logger.debug("[SessionManager] Calling on_emotion_update callback...")
                    self.on_emotion_update({
                        'emotion': filtered_emotion,
                        'target': target_emotion,
                        'deviation': deviation,
                        'session_time': session_time
                    })
            else:
                logger.info("[SessionManager] No fused_emotion available for update.")

        except Exception as e:
            logger.error(f"Error in emotion processing: {e}")

    def _get_current_emotion(self) -> Optional[np.ndarray]:
        """Get current emotion from sensors"""
        try:
            # Try to get emotion from queue first (most recent)
            try:
                emotion_data = self.emotion_data_queue.get_nowait()
                logger.debug(f"[SessionManager] _get_current_emotion got from queue: {emotion_data}")
                return emotion_data
            except queue.Empty:
                logger.debug("[SessionManager] _get_current_emotion: emotion_data_queue empty.")
                pass

            # Fall back to latest emotion data
            with self.emotion_data_lock:
                if self.latest_emotion_data is not None:
                    logger.debug(f"[SessionManager] _get_current_emotion using latest_emotion_data: {self.latest_emotion_data}")
                    return self.latest_emotion_data.copy()

            # No emotion data available
            logger.debug("[SessionManager] _get_current_emotion: No emotion data available.")
            return None

        except Exception as e:
            logger.error(f"Error getting current emotion: {e}")
            return None

    def _normalize_music_parameters(self, params):
        """Normalize music parameter keys to camelCase and ensure all expected keys are present for the frontend."""
        # Map of snake_case to camelCase for frontend
        key_map = {
            'tempo_bpm': 'tempo',
            'rhythm_complexity': 'rhythmComplexity',
            'harmonic_complexity': 'harmonicComplexity',
            'texture_density': 'textureDensity',
            'overall_volume': 'volume',
            'brightness': 'brightness',
            'mode': 'mode',
            'melodic_var': 'melodicVar',
            'harmonic_var': 'harmonicVar',
            # Add more as needed
        }
        # All keys expected by frontend
        expected_keys = [
            'tempo', 'rhythmComplexity', 'harmonicComplexity', 'textureDensity',
            'volume', 'brightness', 'mode', 'melodicVar', 'harmonicVar'
        ]
        normalized = {}
        for k, v in params.items():
            camel = key_map.get(k, k)
            normalized[camel] = v
        for k in expected_keys:
            if k not in normalized:
                normalized[k] = 0
        return normalized

    def _normalize_trajectory_path(self, path):
        """Ensure all trajectory path points are dicts with valence and arousal keys."""
        norm = []
        for pt in path:
            if isinstance(pt, dict):
                v = pt.get('valence', pt.get('mean', {}).get('valence', None))
                a = pt.get('arousal', pt.get('mean', {}).get('arousal', None))
                t = pt.get('timestamp', None)
            elif isinstance(pt, (tuple, list)) and len(pt) >= 2:
                v, a = pt[:2]
                t = pt[2] if len(pt) > 2 else None
            else:
                continue
            if v is not None and a is not None:
                norm.append({'valence': float(v), 'arousal': float(a), 'timestamp': t})
        return norm

    def _get_trajectory_progress(self, session_time=None):
        """Return a robust, always-present trajectory progress structure for frontend visualization."""
        if session_time is None:
            session_time = time.time() - self.session_start_time if self.running else 0
        actual_traj = self.emotion_state.get_emotion_trajectory() or []
        # --- FIX: Sample trajectory function if needed ---
        target_traj = []
        if hasattr(self.trajectory_planner, 'current_trajectory'):
            if callable(self.trajectory_planner.current_trajectory):
                session_duration = self.config.session_duration
                times = np.linspace(0, session_duration, 100)
                target_traj = [
                    self.trajectory_planner.current_trajectory(t) for t in times
                ]
            else:
                target_traj = self.trajectory_planner.current_trajectory
        actual_path = self._normalize_trajectory_path(actual_traj)
        target_path = self._normalize_trajectory_path(target_traj)
        # DTW expects tuples
        actual_dtw = [(pt['valence'], pt['arousal']) for pt in actual_path]
        target_dtw = [(pt['valence'], pt['arousal']) for pt in target_path]
        current_target = self.trajectory_planner.get_current_target() if self.running else None
        deviation = self.dtw_matcher.compute_trajectory_deviation(
            actual_dtw,
            target_dtw,
            self.trajectory_planner.start_time if hasattr(self.trajectory_planner, 'start_time') else 0
        )
        progress = session_time / self.config.session_duration if self.running and self.config.session_duration else 0
        return {
            'actual_path': actual_path,
            'target_path': target_path,
            'current_target': current_target,
            'deviation': deviation,
            'session_time': session_time,
            'progress': progress
        }

    def get_rl_status(self):
        """Return RL and adaptation status for frontend/logging/debugging. Ensures all expected subkeys are present."""
        rl_status = {}
        # Expected subkeys for frontend
        expected_keys = [
            'training_summary', 'feedback', 'adaptation',
            'buffer_size', 'current_params', 'reward', 'confidence',
            'trend', 'total_adaptations', 'recent_adaptations', 'adaptation_rate'
        ]
        if self.rl_agent:
            rl_status['training_summary'] = self.rl_agent.get_training_summary()
        if hasattr(self, 'feedback_processor'):
            rl_status['feedback'] = self.feedback_processor.process_feedback_for_learning()
        if hasattr(self.trajectory_planner, 'get_adaptation_statistics'):
            rl_status['adaptation'] = self.trajectory_planner.get_adaptation_statistics()
        # Fill in all expected keys with safe defaults if missing
        for k in expected_keys:
            if k not in rl_status:
                rl_status[k] = None
        return rl_status

    def get_current_status(self) -> Dict:
        """Get current session status"""
        session_time = time.time() - self.session_start_time if self.running else 0
        with self.emotion_data_lock:
            latest_raw_emotion = self.latest_emotion_data
        if latest_raw_emotion is not None:
            logger.info(f"[get_current_status] latest_raw_emotion type: {type(latest_raw_emotion)}")
        def safe_tolist(obj):
            try:
                return obj.tolist()
            except AttributeError:
                return obj
        trajectory_progress = self._get_trajectory_progress(session_time)
        rl_status = self.get_rl_status()
        # Normalize music parameters for frontend
        music_params = getattr(self, '_latest_music_parameters', self.music_renderer.current_params)
        music_params = self._normalize_music_parameters(music_params)
        return {
            'running': self.running,
            'session_time': session_time,
            'progress': session_time / self.config.session_duration if self.running else 0,
            'current_emotion': self.emotion_state.get_current_emotion(),
            'latest_raw_emotion': safe_tolist(latest_raw_emotion),
            'target_emotion': self.trajectory_planner.get_current_target() if self.running else None,
            'music_parameters': music_params,
            'emotion_capture_running': self.emotion_capture.is_running() if hasattr(self.emotion_capture, 'is_running') else False,
            'queue_size': self.emotion_data_queue.qsize(),
            'trajectory_progress': trajectory_progress,
            'rl_status': rl_status
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
        """Play or re-send the last rendered music structure to the music engine."""
        if self.music_engine:
            params = self.music_renderer.current_params
            music_struct = self.music_renderer.render(params)
            self.music_engine.play(music_struct)

    def pause_music(self):
        """Pause or stop the music engine playback."""
        if self.music_engine:
            self.music_engine.stop()

    def regenerate_music(self):
        """Regenerate music by resetting or randomizing parameters and playing new music."""
        if self.music_engine:
            # Optionally randomize or reset parameters
            params = self.music_renderer.param_space.get_default_parameters()
            self.music_renderer.update_target_parameters(params)
            music_struct = self.music_renderer.render(params)
            self.music_engine.play(music_struct)

    def process_feedback(self, session_id: str, feedback: dict):
        """Process feedback received from the client."""
        if not self.config.enable_feedback:
            logger.warning("Feedback processing is disabled.")
            return
        try:
            # Collect explicit feedback
            self.feedback_collector.collect_explicit_feedback(
                rating=feedback.get('rating', 0),
                context=feedback.get('comments', '')
            )
            # Optionally process implicit feedback
            self.feedback_collector.collect_implicit_feedback(
                interaction_type="feedback_submit",
                intensity=1.0
            )
            logger.info(f"Feedback processed for session {session_id}: {feedback}")
            logger.log_feedback(
                feedback_type='explicit',
                rating=feedback.get('rating', 0),
                category=feedback.get('category', None),
                context={'comments': feedback.get('comments', '')}
            )
        except Exception as e:
            logger.error(f"Error processing feedback for session {session_id}: {e}")
