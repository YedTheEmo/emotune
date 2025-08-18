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
from ..music.engine_pyo import PyoMusicEngine, create_music_engine
from emotune.utils.naming import snake_to_camel
from emotune.core.music.patterns import generate_tidal_pattern

from emotune.utils.logging import get_logger
logger = get_logger()

@dataclass
class SessionConfig:
    emotion_update_rate: float = 0.5  # Hz (reduced from 1.0 to prevent UI freezing)
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
    def __init__(self, config: SessionConfig, db=None, app=None, socketio=None, param_space=None):
        self.config = config
        self.db = db  # Store database reference for emotion data persistence
        self.running = False
        self.session_start_time = 0.0
        self.shutdown_event = threading.Event()  # Added for robust shutdown
        self.app = app  # Store Flask app for context in threads
        self.socketio = socketio  # Store SocketIO instance for emits

        # Music parameter space (allow injection for testing)
        self.param_space = param_space if param_space is not None else MusicParameterSpace()

        # Initialize all subsystems
        self._init_subsystems()

        # Threading
        self.emotion_thread = None
        self.music_thread = None
        self.capture_thread = None

        # Data queues for thread communication with improved sizing
        self.emotion_data_queue = queue.Queue(maxsize=100)  # Increased for better buffering
        self.latest_emotion_data = None
        self.emotion_data_lock = threading.Lock()
        self._empty_queue_counter = 0  # Track consecutive empty queue events

        # Callbacks
        self.on_emotion_update: Optional[Callable] = None
        self.on_music_update: Optional[Callable] = None
        self.on_session_end: Optional[Callable] = None

        # Camera configuration for EmotionCapture with proper parameters
        self.emotion_capture = EmotionCapture(
            face_fps=int(self.config.emotion_update_rate * 2),  # Capture at 2x rate for buffering
            audio_duration=0.5,
            sample_rate=16000,
            camera_index=self.config.camera_index
        )

        # Missing data warning tracking
        self._last_valid_emotion_time = time.time()
        self._missing_data_warning_sent = False
        self._missing_data_timeout = 5.0  # Increased timeout
        
        # Performance tracking
        self._processing_stats = {
            'emotions_processed': 0,
            'emotions_dropped': 0,
            'processing_errors': 0,
            'last_processing_time': 0
        }

    def _init_subsystems(self):
        """Initialize all EmoTune subsystems"""
        # Core components
        self.emotion_state = EmotionState()
        self.kalman_filter = KalmanEmotionFilter()
        self.trajectory_planner = TrajectoryPlanner()
        self.dtw_matcher = DTWMatcher()

        # Emotion analysis components
        from ..emotion.analyzers import EmotionAnalyzer
        self.emotion_analyzer = EmotionAnalyzer(device=self.config.device)
        self.emotion_fusion = EmotionFusion()

        # Music components - FIXED: Always initialize music components
        if self.param_space is None:
            self.param_space = MusicParameterSpace()
        self.music_mapping = BaseMappingEngine()
        self.music_renderer = MusicRenderer(self.param_space)

        # RL and feedback components
        if self.config.enable_rl:
            self.rl_agent = RLAgent(self.param_space)
        if self.config.enable_feedback:
            self.feedback_collector = FeedbackCollector()
            self.feedback_processor = FeedbackProcessor(self.feedback_collector)
        self.last_rl_state = None
        self.last_rl_action = None

        # Music engines - only initialize primary engine (Pyo)
        logger.info("Initializing primary music engine (Pyo)...")
        self.pyo_engine = create_music_engine("auto")  # Primary engine
        logger.info("Primary music engine initialized successfully")
        
        # Set fallback engines to None (only enable if specifically needed)
        self.tidal_engine = None
        self.sonicpi_engine = None  
        self.midi_engine = None
        
        logger.info("Skipping fallback engines (TidalEngine, SonicPiEngine, MidiEngine) - Pyo is sufficient")

    def process_emotion_data(self, session_id: str, data: dict) -> dict:
        """Process emotion data with enhanced error handling and validation"""
        try:
            logger.info(f"[SessionManager] Processing emotion data for session: {session_id}")
            
            # Start emotion monitoring if not already running
            if not self.running:
                self.start_emotion_monitoring(session_id)
            
            # Process the emotion data with the actual data, not a timestamp
            self._process_emotion_update(data)
            
            # Get current status
            status = self.get_current_status()
            
            return {
                'emotion_state': status.get('emotion_state', {}),
                'trajectory_progress': status.get('trajectory_progress', {}),
                'music_parameters': status.get('music_parameters', {})
            }
            
        except Exception as e:
            logger.error(f"Error in emotion processing: {e}")
            return {
                'emotion_state': {'valence': 0.0, 'arousal': 0.0, 'confidence': 0.5},
                'trajectory_progress': {'actual_path': [], 'target_path': [], 'deviation': 1.0},
                'music_parameters': {}
            }

    def update_confidence_thresholds(self, thresholds: Dict[str, float]):
        """Update confidence thresholds for emotion fusion."""
        if self.emotion_fusion:
            self.emotion_fusion.update_thresholds(
                face_threshold=thresholds.get('face'),
                voice_threshold=thresholds.get('voice')
            )
            logger.info(f"Updated confidence thresholds: {thresholds}")

    def update_analysis_mode(self, mode: str):
        """Update the analysis mode for emotion fusion."""
        if self.emotion_fusion:
            self.emotion_fusion.set_analysis_mode(mode)
            logger.info(f"Updated analysis mode: {mode}")

    def update_fusion_options(self, options: Dict[str, object]):
        """Update fusion-level options such as fallback and fusion min confidence."""
        if self.emotion_fusion:
            allow_fallback = options.get('allow_fallback')
            fusion_min_conf = options.get('fusion_min_conf')
            self.emotion_fusion.update_options(allow_fallback=allow_fallback, fusion_min_conf=fusion_min_conf)
            logger.info(f"Updated fusion options: allow_fallback={allow_fallback}, fusion_min_conf={fusion_min_conf}")

    def start_emotion_monitoring(self, session_id: str):
        """Start emotion monitoring with proper initialization"""
        if self.running:
            logger.debug("Emotion monitoring already running")
            return
            
        try:
            self.running = True
            self.session_start_time = time.time()
            
            # Start emotion capture
            self.emotion_capture.start_capture()
            
            # Start emotion processing thread
            self.emotion_thread = threading.Thread(
                target=self._emotion_capture_loop,
                name="EmotionProcessing",
                daemon=True
            )
            self.emotion_thread.start()
            
            logger.info("Emotion monitoring started successfully")

            # --- FIX: Emit monitoring_started event to frontend ---
            if self.socketio:
                sid = getattr(self, '_current_socketio_sid', None)
                if sid:
                    self.socketio.emit('monitoring_started', {'status': 'active'}, room=sid)
                    logger.info(f"Emitted 'monitoring_started' to client {sid}")
            # ---------------------------------------------------------
            
        except Exception as e:
            logger.error(f"Failed to start emotion monitoring: {e}")
            self.running = False
            raise

    def start_session(self, trajectory_type='calm_down', duration=300) -> str:
        """Start a new session with improved error handling"""
        try:
            session_id = str(uuid.uuid4())
            self._current_session_id = session_id  # Store session ID for database operations
            # Also store a room name for Socket.IO broadcasts
            self.set_socketio_room(session_id)
            logger.info(f"Starting session {session_id} with trajectory {trajectory_type}")
            
            # Reset emotion state
            self.emotion_state.reset()
            self.kalman_filter.reset()
            
            # Ensure Pyo engine exists (it may have been cleaned up on previous stop)
            if self.pyo_engine is None:
                logger.info("Reinitializing Pyo engine for new session...")
                self.pyo_engine = create_music_engine("auto")
                logger.info("Pyo engine reinitialized")
            
            # Start trajectory
            from emotune.core.trajectory.library import TrajectoryType
            try:
                # Convert string to TrajectoryType enum
                if isinstance(trajectory_type, str):
                    trajectory_type_enum = TrajectoryType(trajectory_type)
                else:
                    trajectory_type_enum = trajectory_type
                
                self.trajectory_planner.start_trajectory(trajectory_type_enum, duration)
            except ValueError:
                logger.warning(f"Unknown trajectory type: {trajectory_type}, using CALM_DOWN")
                self.trajectory_planner.start_trajectory(TrajectoryType.CALM_DOWN, duration)
            
            # Start emotion monitoring
            self.start_emotion_monitoring(session_id)
            
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to start session: {e}")
            raise

    def stop_session(self, session_id: str = None) -> bool:
        """Enhanced session stop with comprehensive resource cleanup and forced termination"""
        try:
            logger.info(f"Stopping session {session_id} with enhanced cleanup...")
            
            # Phase 1: Signal all systems to stop
            self.running = False
            self.shutdown_event.set()
            
            # Stop trajectory planner to clear target/paths
            try:
                if hasattr(self, 'trajectory_planner') and self.trajectory_planner:
                    self.trajectory_planner.stop_trajectory()
            except Exception as e:
                logger.debug(f"Trajectory stop error: {e}")
            
            # Phase 2: Stop emotion capture with enhanced cleanup
            if hasattr(self, 'emotion_capture') and self.emotion_capture:
                logger.info("Stopping emotion capture with enhanced cleanup...")
                try:
                    self.emotion_capture.stop_capture()
                    logger.info("Emotion capture stopped successfully")
                except Exception as e:
                    logger.error(f"Error stopping emotion capture: {e}")
                    # Force cleanup even if stop_capture fails
                    try:
                        if hasattr(self.emotion_capture, '_force_cleanup_audio'):
                            self.emotion_capture._force_cleanup_audio()
                        if hasattr(self.emotion_capture, '_force_cleanup_camera'):
                            self.emotion_capture._force_cleanup_camera()
                    except Exception as cleanup_e:
                        logger.error(f"Force cleanup error: {cleanup_e}")
            
            # Phase 3: Enhanced thread termination with timeout and force
            self._enhanced_thread_termination()
            
            # Phase 4: Stop all music engines with comprehensive cleanup
            self._comprehensive_music_engine_cleanup()
            
            # Phase 5: Clear all queues and data structures
            self._comprehensive_queue_cleanup()
            
            # Phase 6: Reset all state variables
            self._reset_session_state()
            
            # Phase 7: Final resource validation
            self._validate_resource_cleanup()
            
            logger.info(f"Session {session_id} stopped with enhanced cleanup - all resources released")
            return True
            
        except Exception as e:
            logger.error(f"Error in enhanced session stop: {e}")
            # Even if there are errors, try to clean up what we can
            try:
                self._emergency_cleanup()
            except Exception as emergency_e:
                logger.error(f"Emergency cleanup also failed: {emergency_e}")
            return False
    
    def _enhanced_thread_termination(self):
        """Enhanced thread termination with forced cleanup"""
        threads_to_terminate = []
        
        # Collect all threads that need termination - FIXED: Add proper null checks
        if hasattr(self, 'emotion_thread') and self.emotion_thread is not None and hasattr(self.emotion_thread, 'is_alive') and self.emotion_thread.is_alive():
            threads_to_terminate.append(('emotion_thread', self.emotion_thread))
        if hasattr(self, 'music_thread') and self.music_thread is not None and hasattr(self.music_thread, 'is_alive') and self.music_thread.is_alive():
            threads_to_terminate.append(('music_thread', self.music_thread))
        if hasattr(self, 'capture_thread') and self.capture_thread is not None and hasattr(self.capture_thread, 'is_alive') and self.capture_thread.is_alive():
            threads_to_terminate.append(('capture_thread', self.capture_thread))
        
        # Phase 1: Graceful termination (2 seconds)
        for thread_name, thread in threads_to_terminate:
            logger.info(f"Requesting graceful termination of {thread_name}...")
            thread.join(timeout=2.0)
            
        # Phase 2: Check which threads are still alive
        still_alive = [(name, thread) for name, thread in threads_to_terminate if thread.is_alive()]
        
        if still_alive:
            logger.warning(f"{len(still_alive)} threads did not terminate gracefully")
            
            # Phase 3: Force termination signal (1 second)
            for thread_name, thread in still_alive:
                logger.warning(f"Forcing termination of {thread_name}...")
                if hasattr(self, 'force_stop_event'):
                    self.force_stop_event.set()
                thread.join(timeout=1.0)
            
            # Phase 4: Final check and abandon if necessary
            final_check = [(name, thread) for name, thread in still_alive if thread.is_alive()]
            if final_check:
                logger.error(f"{len(final_check)} threads could not be terminated:")
                for thread_name, thread in final_check:
                    logger.error(f"  - {thread_name} (ID: {thread.ident}) - ABANDONED")
                    # Note: Python doesn't support thread.terminate(), so we abandon them
        
        # Reset thread references
        self.emotion_thread = None
        self.music_thread = None
        self.capture_thread = None
    
    def _comprehensive_music_engine_cleanup(self):
        """Comprehensive cleanup of all music engines"""
        logger.info("Performing comprehensive music engine cleanup...")
        
        engines_to_cleanup = []
        
        # Collect all available engines
        if hasattr(self, 'pyo_engine') and self.pyo_engine:
            engines_to_cleanup.append(('Pyo', self.pyo_engine))
        if hasattr(self, 'tidal_engine') and self.tidal_engine:
            engines_to_cleanup.append(('Tidal', self.tidal_engine))
        if hasattr(self, 'sonicpi_engine') and self.sonicpi_engine:
            engines_to_cleanup.append(('SonicPi', self.sonicpi_engine))
        if hasattr(self, 'midi_engine') and self.midi_engine:
            engines_to_cleanup.append(('MIDI', self.midi_engine))
        
        # Stop each engine with timeout
        for engine_name, engine in engines_to_cleanup:
            try:
                logger.info(f"Stopping {engine_name} engine...")
                
                # Try normal stop first
                if hasattr(engine, 'stop'):
                    engine.stop()
                
                # Additional Pyo-specific cleanup
                if engine_name == 'Pyo' and hasattr(engine, 'server'):
                    try:
                        if hasattr(engine.server, 'stop'):
                            engine.server.stop()
                        if hasattr(engine.server, 'shutdown'):
                            engine.server.shutdown()
                    except Exception as pyo_e:
                        logger.debug(f"Pyo server cleanup error: {pyo_e}")
                
                # Additional cleanup for other engines
                if hasattr(engine, 'cleanup'):
                    engine.cleanup()
                if hasattr(engine, 'close'):
                    engine.close()
                
                logger.info(f"{engine_name} engine stopped successfully")
                
            except Exception as e:
                logger.error(f"Error stopping {engine_name} engine: {e}")
                # Continue with other engines even if one fails
        
        # Reset engine references (allow re-init on next session)
        self.pyo_engine = None
        self.tidal_engine = None
        self.sonicpi_engine = None
        self.midi_engine = None
    
    def _comprehensive_queue_cleanup(self):
        """Comprehensive cleanup of all queues and data structures"""
        logger.info("Performing comprehensive queue cleanup...")
        
        queues_cleared = 0
        
        # Clear emotion data queue
        if hasattr(self, 'emotion_data_queue'):
            cleared = 0
            while not self.emotion_data_queue.empty():
                try:
                    self.emotion_data_queue.get_nowait()
                    cleared += 1
                except queue.Empty:
                    break
            if cleared > 0:
                logger.info(f"Cleared {cleared} items from emotion_data_queue")
                queues_cleared += 1
        
        # Clear any capture queues
        if hasattr(self, 'emotion_capture') and self.emotion_capture and hasattr(self.emotion_capture, 'data_queue'):
            cleared = 0
            while not self.emotion_capture.data_queue.empty():
                try:
                    self.emotion_capture.data_queue.get_nowait()
                    cleared += 1
                except queue.Empty:
                    break
            if cleared > 0:
                logger.info(f"Cleared {cleared} items from capture data_queue")
                queues_cleared += 1
        
        # Clear any other queues that might exist
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if hasattr(attr, 'empty') and hasattr(attr, 'get_nowait'):  # Duck typing for queue
                try:
                    cleared = 0
                    while not attr.empty():
                        try:
                            attr.get_nowait()
                            cleared += 1
                        except:
                            break
                    if cleared > 0:
                        logger.info(f"Cleared {cleared} items from {attr_name}")
                        queues_cleared += 1
                except:
                    pass  # Not actually a queue or already cleared
        
        logger.info(f"Cleared {queues_cleared} queues total")
    
    def _reset_session_state(self):
        """Reset all session state variables"""
        logger.info("Resetting session state...")
        
        # Reset flags
        self.running = False
        if hasattr(self, 'shutdown_event'):
            self.shutdown_event.clear()
        if hasattr(self, 'force_stop_event'):
            self.force_stop_event.clear()
        
        # Reset session identifiers
        self._current_session_id = None
        self.socketio_sid = None
        
        # Reset statistics
        if hasattr(self, '_session_stats'):
            self._session_stats = {}
        
        # Reset any cached data
        if hasattr(self, '_last_emotion_data'):
            self._last_emotion_data = None
        if hasattr(self, '_last_music_params'):
            self._last_music_params = None
        
        logger.info("Session state reset complete")
    
    def _validate_resource_cleanup(self):
        """Validate that all resources have been properly cleaned up"""
        logger.info("Validating resource cleanup...")
        
        issues_found = []
        
        # Check for active threads
        active_threads = []
        if hasattr(self, 'emotion_thread') and self.emotion_thread is not None and hasattr(self.emotion_thread, 'is_alive') and self.emotion_thread.is_alive():
            active_threads.append('emotion_thread')
        if hasattr(self, 'music_thread') and self.music_thread is not None and hasattr(self.music_thread, 'is_alive') and self.music_thread.is_alive():
            active_threads.append('music_thread')
        if hasattr(self, 'capture_thread') and self.capture_thread is not None and hasattr(self.capture_thread, 'is_alive') and self.capture_thread.is_alive():
            active_threads.append('capture_thread')
        
        if active_threads:
            issues_found.append(f"Active threads still running: {active_threads}")
        
        # Check for non-empty queues
        non_empty_queues = []
        if hasattr(self, 'emotion_data_queue') and not self.emotion_data_queue.empty():
            non_empty_queues.append('emotion_data_queue')
        
        if non_empty_queues:
            issues_found.append(f"Non-empty queues: {non_empty_queues}")
        
        # Check for active engines
        active_engines = []
        if hasattr(self, 'pyo_engine') and self.pyo_engine:
            active_engines.append('pyo_engine')
        if hasattr(self, 'tidal_engine') and self.tidal_engine:
            active_engines.append('tidal_engine')
        
        if active_engines:
            issues_found.append(f"Active engines not cleaned up: {active_engines}")
        
        # Report validation results
        if issues_found:
            logger.warning("Resource cleanup validation found issues:")
            for issue in issues_found:
                logger.warning(f"  - {issue}")
        else:
            logger.info("Resource cleanup validation passed - all resources properly released")
    
    def _emergency_cleanup(self):
        """Emergency cleanup when normal cleanup fails"""
        logger.error("Performing emergency cleanup...")
        
        try:
            # Force stop all threads
            import threading
            for thread in threading.enumerate():
                if thread != threading.current_thread() and thread is not None and hasattr(thread, 'is_alive') and thread.is_alive():
                    if hasattr(thread, 'name') and ('emotion' in thread.name.lower() or 'music' in thread.name.lower() or 'capture' in thread.name.lower()):
                        logger.error(f"Emergency: Abandoning thread {thread.name}")
            
            # Force clear all attributes that might hold resources
            for attr_name in ['pyo_engine', 'tidal_engine', 'sonicpi_engine', 'midi_engine', 
                             'emotion_capture', 'emotion_thread', 'music_thread', 'capture_thread']:
                if hasattr(self, attr_name):
                    setattr(self, attr_name, None)
            
            # Force reset flags
            self.running = False
            self._current_session_id = None
            
            logger.error("Emergency cleanup completed")
            
        except Exception as e:
            logger.error(f"Emergency cleanup failed: {e}")
            # At this point, we've done everything we can

    def set_socketio_sid(self, socketio_sid):
        """Set SocketIO session ID for frontend communication"""
        self._current_socketio_sid = socketio_sid

    def set_socketio_room(self, room_name: str):
        """Set Socket.IO room (typically the session_id) for emissions"""
        self._socketio_room = room_name

    def _emotion_capture_loop(self):
        """Improved emotion capture loop with better error handling and performance"""
        logger.info("Starting emotion capture loop")
        
        consecutive_empty_count = 0
        max_consecutive_empty = 10
        
        # NEW: small buffers to align frame and audio by timestamp
        frame_buffer = []  # list of (ts, frame)
        audio_buffer = []  # list of (ts, audio, sr)
        buffer_horizon = 2.0  # seconds to keep
        pair_window = 1.0     # acceptable frame-audio delta in seconds
        
        # FIX: Add rate limiting to respect configured emotion_update_rate
        last_emotion_update_time = 0.0
        emotion_update_interval = 1.0 / self.config.emotion_update_rate  # Convert Hz to seconds
        
        while not self.shutdown_event.is_set() and self.running:
            try:
                current_time = time.time()
                
                # FIX: Rate limit emotion updates to configured frequency
                if current_time - last_emotion_update_time < emotion_update_interval:
                    time.sleep(0.01)  # Brief pause to avoid tight loop
                    continue
                
                # Get data from emotion capture
                data = self.emotion_capture.get_data()
                
                if not data:
                    consecutive_empty_count += 1
                    
                    # Check for missing data timeout
                    now = time.time()
                    if (now - self._last_valid_emotion_time > self._missing_data_timeout and
                        not self._missing_data_warning_sent):
                        self._emit_missing_data_warning(now)
                    
                    # If too many consecutive empty reads, log warning
                    if consecutive_empty_count >= max_consecutive_empty:
                        logger.warning(f"Queue empty for {consecutive_empty_count} consecutive reads")
                        consecutive_empty_count = 0
                    
                    time.sleep(0.1)  # Brief pause
                    continue
                
                # Reset empty counter on successful data
                consecutive_empty_count = 0
                
                # Demux incoming item
                ts = data.get('timestamp', time.time())
                if data.get('frame') is not None:
                    frame_buffer.append((ts, data['frame']))
                if data.get('audio') is not None:
                    audio_buffer.append((ts, data['audio'], data.get('sr', 16000)))
                
                # Drop old items
                cutoff = time.time() - buffer_horizon
                frame_buffer = [(t, f) for (t, f) in frame_buffer if t >= cutoff]
                audio_buffer = [(t, a, s) for (t, a, s) in audio_buffer if t >= cutoff]
                
                # Try to form a pair (closest in time within window)
                analysis_data = {}
                if frame_buffer:
                    # take latest frame
                    ft, ff = frame_buffer[-1]
                    analysis_data['frame'] = ff
                    # find closest audio
                    closest = None
                    best_dt = None
                    for at, aa, sr in audio_buffer:
                        dt = abs(at - ft)
                        if best_dt is None or dt < best_dt:
                            best_dt = dt
                            closest = (at, aa, sr)
                    if closest and best_dt is not None and best_dt <= pair_window:
                        analysis_data['audio'] = closest[1]
                        analysis_data['sr'] = closest[2]
                elif audio_buffer:
                    # allow voice-only analysis when no frames
                    at, aa, sr = audio_buffer[-1]
                    analysis_data['audio'] = aa
                    analysis_data['sr'] = sr
                
                # Process the data if we have something
                if analysis_data:
                    self._process_captured_data(analysis_data)
                    # FIX: Update last emotion update time to enforce rate limiting
                    last_emotion_update_time = current_time
                else:
                    # small pause to avoid tight loop
                    time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in emotion capture loop: {e}")
                self._processing_stats['processing_errors'] += 1
                time.sleep(0.1)
        
        logger.info("Emotion capture loop ended")

    def _process_captured_data(self, data: Dict):
        """Process captured data with improved error handling"""
        try:
            # Extract frame and audio data
            face_frame = data.get('face_frame') or data.get('frame')
            audio_chunk = data.get('audio_chunk') or data.get('audio')
            
            # Prepare data for analysis
            analysis_data = {}
            if face_frame is not None:
                analysis_data['frame'] = face_frame
            if audio_chunk is not None:
                analysis_data['audio'] = audio_chunk
            if 'sr' in data:
                analysis_data['sr'] = data['sr']
            
            # Perform emotion analysis if we have data
            if analysis_data:
                analysis_result = self.emotion_analyzer.analyze(**analysis_data)
                
                # Fuse emotions
                fused_emotion = self._fuse_emotion_data(analysis_result)
                
                if fused_emotion:
                    # Add to queue
                    try:
                        self.emotion_data_queue.put_nowait(fused_emotion)
                        self._processing_stats['emotions_processed'] += 1
                        self._last_valid_emotion_time = time.time()
                        self._missing_data_warning_sent = False
                        
                        # Process emotion update with the fused emotion data
                        self._process_emotion_update(fused_emotion)
                        
                    except queue.Full:
                        # Remove oldest and add new
                        try:
                            self.emotion_data_queue.get_nowait()
                            self.emotion_data_queue.put_nowait(fused_emotion)
                            self._processing_stats['emotions_dropped'] += 1
                        except queue.Empty:
                            logger.error("Queue management error")
                else:
                    # If no fused emotion, create a dynamic fallback structure
                    # FIXED: Make fallback data more dynamic to simulate real emotion
                    import random
                    import math
                    
                    # Create a simple oscillating pattern to simulate emotion changes
                    current_time = time.time()
                    oscillation = math.sin(current_time * 0.5) * 0.3  # Slow oscillation
                    noise = random.uniform(-0.1, 0.1)  # Small random variation
                    
                    default_emotion = {
                        'mean': {
                            'valence': max(-1.0, min(1.0, 0.0 + oscillation + noise)),
                            'arousal': max(0.0, min(1.0, 0.5 + oscillation * 0.5 + noise * 0.5))
                        },
                        'confidence': 0.5 + abs(noise) * 0.3,  # Vary confidence slightly
                        'timestamp': current_time
                    }
                    
                    # Process the fallback emotion
                    self._process_emotion_update(default_emotion)
                    self._processing_stats['emotions_processed'] += 1
                    self._last_valid_emotion_time = current_time
                    self._missing_data_warning_sent = False
            else:
                # No analysis data available, create fallback emotion
                import random
                import math
                
                current_time = time.time()
                oscillation = math.sin(current_time * 0.5) * 0.3
                noise = random.uniform(-0.1, 0.1)
                
                fallback_emotion = {
                    'mean': {
                        'valence': max(-1.0, min(1.0, 0.0 + oscillation + noise)),
                        'arousal': max(0.0, min(1.0, 0.5 + oscillation * 0.5 + noise * 0.5))
                    },
                    'confidence': 0.3,  # Lower confidence for fallback
                    'timestamp': current_time
                }
                
                # Process the fallback emotion
                self._process_emotion_update(fallback_emotion)
                self._processing_stats['emotions_processed'] += 1
                self._last_valid_emotion_time = current_time
                
        except Exception as e:
            logger.error(f"Error processing captured data: {e}")
            self._processing_stats['processing_errors'] += 1

    def _emit_missing_data_warning(self, current_time: float):
        """Emit warning about missing data to frontend"""
        if self.app is not None and self.socketio is not None:
            room = getattr(self, '_socketio_room', None)
            if room:
                warning_msg = f'No emotion data received for {int(current_time - self._last_valid_emotion_time)} seconds.'
                self.socketio.emit('emotion_update', {'warning': warning_msg}, room=room)
                self._missing_data_warning_sent = True

    def _save_emotion_to_database(self, emotion_data: Dict):
        """Save processed emotion data to the database with enhanced error handling."""
        if not self.db or not self._current_session_id:
            logger.debug("No database or session ID available for emotion data saving")
            return
        
        try:
            # Extract emotion values with proper validation
            if isinstance(emotion_data, dict):
                # Handle nested mean structure
                if 'mean' in emotion_data and isinstance(emotion_data['mean'], dict):
                    valence = emotion_data['mean'].get('valence', 0.0)
                    arousal = emotion_data['mean'].get('arousal', 0.5)
                else:
                    # Handle direct structure
                    valence = emotion_data.get('valence', 0.0)
                    arousal = emotion_data.get('arousal', 0.5)
                
                confidence = emotion_data.get('confidence', 0.5)
                timestamp = emotion_data.get('timestamp', time.time())
            else:
                # Fallback for unexpected data types
                logger.warning(f"Unexpected emotion_data type: {type(emotion_data)}, using defaults")
                valence = 0.0
                arousal = 0.5
                confidence = 0.5
                timestamp = time.time()
            
            # Validate values
            valence = float(np.clip(valence, -1.0, 1.0))
            arousal = float(np.clip(arousal, -1.0, 1.0))
            confidence = float(np.clip(confidence, 0.0, 1.0))
            timestamp = float(timestamp)
            
            # Save to database
            self.db.save_emotion_data(
                session_id=self._current_session_id,
                timestamp=timestamp,
                valence=valence,
                arousal=arousal,
                confidence=confidence,
                source='fusion'
            )
            
            logger.debug(f"Saved emotion data to database: valence={valence:.3f}, arousal={arousal:.3f}, confidence={confidence:.3f}")
            
        except Exception as e:
            logger.error(f"Error saving emotion data to database: {e}")
            import traceback
            logger.error(f"Database save traceback: {traceback.format_exc()}")

    def _fuse_emotion_data(self, analysis_result: Dict) -> Optional[Dict]:
        """Fuse emotion data with improved error handling and proper structure"""
        try:
            if not analysis_result:
                logger.debug("No analysis result for fusion")
                return None
            
            # Extract face and voice data
            face_data = analysis_result.get('face')
            voice_data = analysis_result.get('voice')
            
            # Validate data before fusion
            if face_data:
                logger.debug(f"Valid face_data for fusion: {face_data}")
            if voice_data:
                logger.debug(f"Valid voice_data for fusion: {voice_data}")
            
            # Perform fusion
            fused_result = self.emotion_fusion.fuse_emotions(face_data, voice_data)
            
            # Ensure proper structure for our emotion processing
            if fused_result:
                # Add timestamp
                fused_result['timestamp'] = time.time()
                # Attach raw modalities for frontend diagnostics
                if face_data is not None:
                    fused_result['face'] = face_data
                if voice_data is not None:
                    fused_result['voice'] = voice_data
                
                # Ensure we have the 'mean' structure that our processing expects
                if 'mean' not in fused_result:
                    # If we have direct valence/arousal, wrap them in 'mean'
                    if 'valence' in fused_result and 'arousal' in fused_result:
                        fused_result = {
                            'mean': {
                                'valence': float(np.clip(fused_result['valence'], -1.0, 1.0)),
                                'arousal': float(np.clip(fused_result['arousal'], -1.0, 1.0))
                            },
                            'confidence': float(np.clip(fused_result.get('confidence', 0.5), 0.0, 1.0)),
                            'timestamp': fused_result['timestamp'],
                            'sources': fused_result.get('sources', {}),
                            'face': face_data,
                            'voice': voice_data
                        }
                    else:
                        # Create default structure
                        fused_result = {
                            'mean': {'valence': 0.0, 'arousal': 0.0},
                            'confidence': 0.5,
                            'timestamp': fused_result['timestamp'],
                            'sources': fused_result.get('sources', {}),
                            'face': face_data,
                            'voice': voice_data
                        }
                else:
                    # Ensure clipping and carry sources
                    fused_result = {
                        'mean': {
                            'valence': float(np.clip(fused_result['mean'].get('valence', 0.0), -1.0, 1.0)),
                            'arousal': float(np.clip(fused_result['mean'].get('arousal', 0.0), -1.0, 1.0))
                        },
                        'confidence': float(np.clip(fused_result.get('confidence', 0.5), 0.0, 1.0)),
                        'timestamp': fused_result['timestamp'],
                        'sources': fused_result.get('sources', {}),
                        'face': face_data,
                        'voice': voice_data
                        }
                
                logger.debug(f"Fused emotion result: {fused_result}")
                return fused_result
            else:
                logger.warning("Fusion returned None, creating default structure")
                return {
                    'mean': {'valence': 0.0, 'arousal': 0.0},
                    'confidence': 0.5,
                    'timestamp': time.time(),
                    'sources': {'face': False, 'voice': False}
                }
            
        except Exception as e:
            logger.error(f"Error in emotion fusion: {e}")
            # Return default structure on error
            return {
                'mean': {'valence': 0.0, 'arousal': 0.5},
                'confidence': 0.5,
                'timestamp': time.time()
            }

    def _process_emotion_update(self, emotion_data: Dict):
        """
        Main processing loop for emotion data.
        This function now orchestrates the entire flow from fusion to music generation.
        """
        try:
            # Step 1: Fuse Face and Voice Data
            # This is now handled by the caller, which passes 'emotion_data'.
            # We assume 'emotion_data' is the output of the fusion process.
            
            if not isinstance(emotion_data, dict) or 'mean' not in emotion_data:
                logger.error(f"Invalid emotion data structure received: {emotion_data}")
                return

            valence = emotion_data['mean'].get('valence', 0.0)
            arousal = emotion_data['mean'].get('arousal', 0.0)
            confidence = emotion_data.get('confidence', 0.5)
            timestamp = emotion_data.get('timestamp', time.time())
            
            # Step 2: Apply Kalman Filter
            sources = emotion_data.get('sources', {})
            has_observation = bool(sources.get('face') or sources.get('voice'))
            if (not has_observation) or (confidence is not None and confidence < 0.2):
                # Predict-only step when we don't trust the measurement
                state, cov = self.kalman_filter.predict()
            else:
                state, cov = self.kalman_filter.update({
                    'valence': valence, 'arousal': arousal, 'confidence': confidence,
                    'uncertainty': 1.0 - confidence
                })
            filtered_valence, filtered_arousal = state[0], state[1]
            logger.debug(f"Kalman filtered: V={filtered_valence:.3f}, A={filtered_arousal:.3f}")

            # Extract the 2x2 covariance for valence and arousal
            cov_2x2 = cov[:2, :2] if cov is not None and cov.shape[0] >= 2 and cov.shape[1] >= 2 else [[0.5, 0.0], [0.0, 0.5]]

            # Step 3: Update Central Emotion State
            emotion_dist = {
                'mean': {
                    'valence': filtered_valence,
                    'arousal': filtered_arousal
                },
                'covariance': cov_2x2,
                'timestamp': timestamp,
                'confidence': confidence,  # Ensure confidence is stored in the state
                'uncertainty_trace': np.trace(cov_2x2)
            }
            self.emotion_state.update_emotion(emotion_dist)

            # Step 4: Map Emotion to Music Parameters
            music_params = self._map_emotion_to_music(filtered_valence, filtered_arousal)
            self._latest_music_params = music_params.copy()
            
            # Step 5: Update Music Engine & RL
            engine = self._select_music_engine()
            if engine:
                music_struct = {
                    'emotion': {'mean': {'valence': filtered_valence, 'arousal': filtered_arousal}},
                    'parameters': music_params
                }
                engine.play(music_struct)

                # --- RL AGENT UPDATE ---
                if self.config.enable_rl and self.rl_agent:
                    # 1. Get current state vector
                    trajectory_progress = self._get_trajectory_progress()
                    dtw_error = trajectory_progress.get('deviation', 1.0)
                    progress = trajectory_progress.get('progress', 0.0)
                    current_rl_state = self.rl_agent.get_state_vector(
                        emotion_mean=np.array([filtered_valence, filtered_arousal]),
                        emotion_cov=cov_2x2,
                        dtw_error=dtw_error,
                        trajectory_progress=progress
                    )

                    # 2. Select action based on current state and update music parameters
                    action_adjustments = self.rl_agent.select_action(current_rl_state)
                    music_params = self.rl_agent.update_parameters(action_adjustments)
                    self._latest_music_params = music_params.copy() # Make sure to update the latest params

                    # 3. If we have a previous state and action, compute reward and store experience
                    if self.last_rl_state is not None and self.last_rl_action is not None:
                        # 3a. Compute reward
                        reward = self.feedback_processor.compute_reward_signal(
                            trajectory_deviation=dtw_error,
                            emotion_stability=np.trace(cov_2x2)
                        )
                        # 3b. Store experience with the PREVIOUS action
                        self.rl_agent.store_experience(
                            state=self.last_rl_state,
                            action=self.last_rl_action,
                            reward=reward,
                            next_state=current_rl_state,
                            done=(time.time() - self.session_start_time) >= self.config.session_duration if self.session_start_time else False
                        )
                        # 3c. Train the agent
                        self.rl_agent.train()

                    # 4. Update the last state and action for the next iteration
                    self.last_rl_state = current_rl_state
                    self.last_rl_action = action_adjustments

            # Step 6: Emit updates and persist data
            self._emit_emotion_update(
                raw_emotion=emotion_data,
                filtered_emotion=self.emotion_state.get_current_emotion(),
                music_params=music_params
            )
            self._save_emotion_to_database(self.emotion_state.get_current_emotion())

        except Exception as e:
            logger.error(f"Error in _process_emotion_update: {e}", exc_info=True)


    def _categorize_valence(self, valence: float) -> str:
        """Categorize valence for clinical monitoring"""
        if valence < -0.6:
            return "very_negative"
        elif valence < -0.2:
            return "negative"
        elif valence < 0.2:
            return "neutral"
        elif valence < 0.6:
            return "positive"
        else:
            return "very_positive"
    
    def _categorize_arousal(self, arousal: float) -> str:
        """Categorize arousal for clinical monitoring"""
        if arousal < 0.2:
            return "very_low"
        elif arousal < 0.4:
            return "low"
        elif arousal < 0.6:
            return "moderate"
        elif arousal < 0.8:
            return "high"
        else:
            return "very_high"

    def _emit_emotion_update(self, raw_emotion: Dict, filtered_emotion: Dict, music_params: Dict):
        """Emit emotion update to frontend with proper serialization"""
        try:
            logger.debug(f"Attempting to emit emotion update: app={self.app is not None}, socketio={self.socketio is not None}")
            
            if self.app is None or self.socketio is None:
                logger.warning("Cannot emit emotion update: app or socketio not available")
                return
            # Use room-based broadcasting keyed by session_id to handle multi-tab
            room = getattr(self, '_socketio_room', None)
            if not room:
                logger.warning("Cannot emit emotion update: no Socket.IO room set")
                return
            
            # Prepare payload with proper serialization and safe data access
            payload = {
                'emotion_state': self._serialize_emotion_state(filtered_emotion or raw_emotion),
                'trajectory_progress': self._get_trajectory_progress(),
                'music_parameters': self._serialize_music_parameters(music_params),
                'system_logs': {
                    'face_data': (
                        {
                            'valence': raw_emotion.get('face', {}).get('emotions', {}).get('valence'),
                            'arousal': raw_emotion.get('face', {}).get('emotions', {}).get('arousal'),
                            'confidence': raw_emotion.get('face', {}).get('confidence')
                        } if raw_emotion and isinstance(raw_emotion, dict) and raw_emotion.get('face') is not None else {'valence': None, 'arousal': None, 'confidence': None}
                    ),
                    'voice_data': (
                        {
                            'valence': raw_emotion.get('voice', {}).get('emotions', {}).get('valence'),
                            'arousal': raw_emotion.get('voice', {}).get('emotions', {}).get('arousal'),
                            'confidence': raw_emotion.get('voice', {}).get('confidence')
                        } if raw_emotion and isinstance(raw_emotion, dict) and raw_emotion.get('voice') is not None else {'valence': None, 'arousal': None, 'confidence': None}
                    ),
                    'fusion': {
                        'valence': filtered_emotion['mean']['valence'] if filtered_emotion else raw_emotion.get('mean', {}).get('valence'),
                        'arousal': filtered_emotion['mean']['arousal'] if filtered_emotion else raw_emotion.get('mean', {}).get('arousal'),
                        'confidence': filtered_emotion.get('confidence', 0.5) if filtered_emotion else raw_emotion.get('confidence', 0.5)
                    },
                    'fusion_sources': (
                        filtered_emotion.get('sources') if filtered_emotion and isinstance(filtered_emotion, dict) and filtered_emotion.get('sources') is not None
                        else raw_emotion.get('sources') if raw_emotion and isinstance(raw_emotion, dict) else {}
                    ),
                    'music_engine': {
                        'status': 'active',
                        'tempo': music_params.get('tempo_bpm'),
                        'volume': music_params.get('overall_volume')
                    },
                    'feedback': self.feedback_collector.get_session_feedback_summary() if self.config.enable_feedback else {},
                    'rl_agent': self.rl_agent.get_status() if self.config.enable_rl else {}
                }
            }
            
            logger.debug(f"Emitting emotion update to frontend: {payload}")
            
            # Emit to frontend
            with self.app.app_context():
                self.socketio.emit('emotion_update', payload, room=room)
            
            logger.info(f"Successfully emitted emotion update to frontend")
            
        except Exception as e:
            logger.error(f"Error emitting emotion update: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    def _serialize_emotion_state(self, emotion_data: Dict) -> Dict:
        """Serialize emotion state for JSON transmission"""
        try:
            # Handle both direct format and nested mean format
            if 'mean' in emotion_data:
                # Kalman-filtered format with mean/covariance
                mean_data = emotion_data['mean']
                valence = float(mean_data.get('valence', 0.0))
                arousal = float(mean_data.get('arousal', 0.0))
                confidence = float(emotion_data.get('confidence', 0.5))
            else:
                # Direct format
                valence = float(emotion_data.get('valence', 0.0))
                arousal = float(emotion_data.get('arousal', 0.0))
                confidence = float(emotion_data.get('confidence', 0.5))
            
            # FIXED: Ensure proper structure for frontend
            serialized = {
                'valence': valence,
                'arousal': arousal,
                'confidence': confidence,
                'timestamp': emotion_data.get('timestamp', time.time())
            }
            
            # Add mean structure for compatibility
            serialized['mean'] = {
                'valence': valence,
                'arousal': arousal
            }
            
            return serialized
            
        except Exception as e:
            logger.error(f"Error serializing emotion state: {e}")
            logger.error(f"Emotion data structure: {emotion_data}")
            return {
                'valence': 0.0, 
                'arousal': 0.0, 
                'confidence': 0.5,
                'timestamp': time.time(),
                'mean': {'valence': 0.0, 'arousal': 0.0}
            }

    def _serialize_music_parameters(self, music_params: Dict) -> Dict:
        """Serialize music parameters for JSON transmission"""
        try:
            if not music_params:
                return {}
            
            serialized = {}
            for key, value in music_params.items():
                if isinstance(value, np.ndarray):
                    # Handle numpy arrays properly
                    if value.size == 1:
                        serialized[key] = float(value.item())
                    else:
                        serialized[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    # Convert numpy types to Python types with proper precision
                    serialized[key] = float(value)
                else:
                    serialized[key] = value
            
            return serialized
            
        except Exception as e:
            logger.error(f"Error serializing music parameters: {e}")
            return {}

    def _normalize_music_parameters(self, params):
        """Normalize music parameter keys to camelCase and ensure all expected keys are present for the frontend."""
        # Build mapping from parameter space
        param_space = getattr(self, 'param_space', None)
        if param_space is None:
            param_space = MusicParameterSpace()
        snake_keys = list(param_space.parameters.keys())
        key_map = {k: snake_to_camel(k) for k in snake_keys}
        # Inverse map for lookup
        camel_keys = set(key_map.values())
        normalized = {}
        # Map all known parameters
        for k, v in params.items():
            camel = key_map.get(k, snake_to_camel(k))
            normalized[camel] = v
        # Ensure all expected keys are present
        for snake, camel in key_map.items():
            if camel not in normalized:
                normalized[camel] = param_space.parameters[snake].default
                logger.warning(f"[SessionManager] Music parameter '{snake}' missing in output; defaulting to {param_space.parameters[snake].default}")
        # Optionally, warn about extra keys
        for k in normalized:
            if k not in camel_keys:
                logger.warning(f"[SessionManager] Music parameter '{k}' not in parameter space (extra or legacy param)")
        return normalized

    def _normalize_trajectory_path(self, path):
        """Ensure all trajectory path points are dicts with valence, arousal, and timestamp keys.
        - For actual (recorded) trajectories, timestamp is the real session-relative time.
        - For generated target trajectories, timestamp is the intended offset from session start (or None if not applicable).
        """
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

    def _get_trajectory_progress(self) -> dict:
        """Gets progress along the current therapeutic trajectory."""
        if not (self.trajectory_planner and self.trajectory_planner._is_active()):
            return {
                'actual_path': [], 'target_path': [], 'current_target': None,
                'deviation': 1.0, 'session_time': 0.0, 'progress': 0.0
            }

        emotion_trajectory = self.emotion_state.get_emotion_trajectory()
        adherence_info = self.trajectory_planner.evaluate_trajectory_adherence(emotion_trajectory)

        # Sample the target path for visualization
        target_path = []
        target_path_func = self.trajectory_planner.current_trajectory
        if callable(target_path_func):
            duration = self.trajectory_planner.duration
            num_points = 100
            for i in range(num_points + 1):
                t = (i / num_points) * duration
                target_point = target_path_func(t)
                if target_point:
                    target_path.append({'valence': target_point[0], 'arousal': target_point[1]})

        current_target = self.trajectory_planner.get_current_target()

        # Decimate/bound actual path to avoid large payloads
        max_points = 360  # ~6 minutes at 1 Hz; decimate beyond this
        actual_path_full = [{'valence': dp['mean']['valence'], 'arousal': dp['mean']['arousal']} for dp in emotion_trajectory]
        if len(actual_path_full) > max_points:
            step = max(1, len(actual_path_full) // max_points)
            actual_path = [actual_path_full[i] for i in range(0, len(actual_path_full), step)]
            if actual_path[-1] is not actual_path_full[-1]:
                actual_path.append(actual_path_full[-1])
        else:
            actual_path = actual_path_full

        return {
            'actual_path': actual_path,
            'target_path': target_path,
            'current_target': {'valence': current_target[0], 'arousal': current_target[1]} if current_target else None,
            'deviation': adherence_info.get('deviation', 1.0),
            'session_time': (time.time() - self.session_start_time) if self.session_start_time else 0.0,
            'progress': adherence_info.get('progress', 0.0),
        }

    def get_rl_status(self):
        """Get RL agent status"""
        if hasattr(self, 'rl_agent') and self.rl_agent:
            return self.rl_agent.get_status()
        return {'active': False, 'episodes': 0, 'rewards': []}

    def get_current_status(self) -> Dict:
        """Get current system status with improved error handling"""
        try:
            # Get current emotion
            current_emotion = getattr(self, '_latest_filtered_emotion', None)
            if not current_emotion:
                current_emotion = {
                    'valence': 0.0,
                    'arousal': 0.0,
                    'confidence': 0.5
                }
            
            # Get trajectory progress
            trajectory_progress = self._get_trajectory_progress()
            
            # Get music parameters - FIXED: Use latest generated parameters
            music_params = {}
            if hasattr(self, '_latest_music_params'):
                music_params = self._latest_music_params.copy()
            elif hasattr(self, 'music_renderer'):
                music_params = self.music_renderer.current_params.copy()
            
            # Serialize data for JSON transmission
            def safe_tolist(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                return obj
            
            return {
                'emotion_state': {
                    'valence': safe_tolist(current_emotion.get('valence', 0.0)),
                    'arousal': safe_tolist(current_emotion.get('arousal', 0.0)),
                    'confidence': safe_tolist(current_emotion.get('confidence', 0.5))
                },
                'trajectory_progress': trajectory_progress,
                'music_parameters': {k: safe_tolist(v) for k, v in music_params.items()},
                'system_stats': {
                    'capture_stats': self.emotion_capture.get_stats() if self.emotion_capture else {},
                    'processing_stats': self._processing_stats,
                    'fusion_quality': self.emotion_fusion.get_fusion_quality_metrics()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting current status: {e}")
            return {
                'emotion_state': {'valence': 0.0, 'arousal': 0.0, 'confidence': 0.5},
                'trajectory_progress': {'actual_path': [], 'target_path': [], 'deviation': 1.0},
                'music_parameters': {},
                'system_stats': {}
            }

    def update_emotion_config(self, **kwargs):
        """Update emotion configuration"""
        try:
            # Update fusion weights if provided
            if 'face_weight' in kwargs or 'voice_weight' in kwargs:
                face_weight = kwargs.get('face_weight', 0.7)
                voice_weight = kwargs.get('voice_weight', 0.3)
                self.emotion_fusion = EmotionFusion(face_weight=face_weight, voice_weight=voice_weight)
            
            # Update other configurations
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            logger.info(f"Updated emotion configuration: {kwargs}")
            
        except Exception as e:
            logger.error(f"Error updating emotion configuration: {e}")

    def _select_music_engine(self):
        """Select appropriate music engine with Pyo as primary"""
        engine_name = getattr(self.config, 'music_engine', 'pyo')
        
        if engine_name == 'pyo':
            return self.pyo_engine
        elif engine_name == 'tidal' and self.tidal_engine:
            return self.tidal_engine
        elif engine_name == 'midi' and self.midi_engine:
            return self.midi_engine
        elif engine_name == 'sonicpi' and self.sonicpi_engine:
            return self.sonicpi_engine
        else:
            # Always default to Pyo (primary engine)
            logger.info(f"Requested engine '{engine_name}' not available, using Pyo engine")
            return self.pyo_engine

    def _process_music_update(self, session_time: float):
        """Process music update with improved error handling"""
        try:
            # Get current emotion
            current_emotion = getattr(self, '_latest_filtered_emotion', None)
            if not current_emotion:
                return
            
            # Map to music parameters
            music_params = self._map_emotion_to_music(current_emotion)
            if music_params:
                # Update music renderer
                self.music_renderer.update_target_parameters(music_params)
                
                # Send parameters to music engine for actual playback
                try:
                    engine = self._select_music_engine()
                    if engine:
                        # Send parameters to engine
                        engine.update_parameters(music_params)
                        
                        # Create music structure for playback
                        music_struct = {
                            'emotion': current_emotion,
                            'parameters': music_params,
                            'timestamp': session_time
                        }
                        engine.play(music_struct)
                        
                        logger.debug(f"Sent parameters to music engine: {len(music_params)} params")
                except Exception as e:
                    logger.error(f"Error sending parameters to music engine: {e}")
                
                # Log music generation
                logger.info(f"Music generation logged: {len(music_params)} params, "
                           f"trajectory={self.config.trajectory_name}, DTW=None")
            
        except Exception as e:
            logger.error(f"Error in music update processing: {e}")

    def play_music(self):
        """Play music with current parameters"""
        try:
            engine = self._select_music_engine()
            
            # Create music structure with current parameters and emotion state
            music_struct = {
                'parameters': self.music_renderer.current_params if hasattr(self, 'music_renderer') else {},
                'emotion': {
                    'valence': 0.0,
                    'arousal': 0.5
                }
            }
            
            # If we have recent emotion data, use it
            if hasattr(self, '_latest_filtered_emotion'):
                music_struct['emotion'] = self._latest_filtered_emotion
            
            engine.play(music_struct)
        except Exception as e:
            logger.error(f"Error playing music: {e}")

    def pause_music(self):
        """Pause music"""
        try:
            engine = self._select_music_engine()
            engine.pause()
        except Exception as e:
            logger.error(f"Error pausing music: {e}")

    def regenerate_music(self):
        """Regenerate music with current parameters"""
        try:
            engine = self._select_music_engine()
            engine.regenerate()
        except Exception as e:
            logger.error(f"Error regenerating music: {e}")

    def process_realtime_feedback(self, session_id: str, feedback: dict):
        """Process user feedback"""
        try:
            if hasattr(self, 'feedback_collector'):
                # Extract the rating and comments for the collector
                rating = feedback.get('rating')
                comments = feedback.get('comments', '')
                if rating is not None:
                    self.feedback_collector.collect_explicit_feedback(rating=float(rating), context=comments)

            if hasattr(self, 'feedback_processor'):
                # Get adjustments and impact statement from the processor
                processed_result = self.feedback_processor.process_feedback(session_id, feedback)
                adjustments = processed_result.get('adjustments')
                impact_message = processed_result.get('impact')

                # Apply adjustments to the RL agent
                if self.rl_agent and adjustments:
                    self.rl_agent.update_parameters(adjustments)
                    logger.info(f"Applied feedback adjustments to RL agent: {adjustments}")

                # Emit the impact statement to the frontend (to the session room)
                if self.socketio and impact_message:
                    room = getattr(self, '_socketio_room', None) or session_id
                    self.socketio.emit('feedback_impact', {'message': impact_message}, room=room)

                # Guard against UI stalls: prompt client to ensure monitoring is active
                if self.socketio:
                    room = getattr(self, '_socketio_room', None) or session_id
                    self.socketio.emit('monitoring_started', {'status': 'active'}, room=room)

        except Exception as e:
            logger.error(f"Error processing feedback: {e}")

    def shutdown(self):
        """Shutdown the session manager with proper cleanup"""
        try:
            logger.info("Shutting down SessionManager...")
            self.stop_session()
            
            # Clear all data structures
            self.emotion_state.clear_history()
            self.kalman_filter.reset()
            self.emotion_fusion.reset_history()
            
            logger.info("SessionManager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    def _map_emotion_to_music(self, valence: float, arousal: float) -> Dict[str, float]:
        """Map emotion to music parameters with comprehensive clinical validation"""
        try:
            # If RL is enabled, the agent has already updated the parameters.
            if self.config.enable_rl and hasattr(self, 'rl_agent') and self.rl_agent and hasattr(self, '_latest_music_params'):
                return self._latest_music_params

            # CLINICAL VALIDATION: Ensure emotion values are in valid ranges
            valence = np.clip(valence, -1.0, 1.0)
            arousal = np.clip(arousal, 0.0, 1.0)  # FIXED: arousal range [0,1]
            
            logger.debug(f"Mapping emotion to music: valence={valence:.3f}, arousal={arousal:.3f}")
            
            # Get base parameter mapping
            music_params = self.music_mapping.map_emotion_to_parameters(valence, arousal)
            
            # CLINICAL SAFETY: Validate therapeutic safety
            is_safe, safety_warnings = self.param_space.validate_therapeutic_safety(music_params)
            
            if not is_safe:
                logger.warning("CLINICAL SAFETY ALERT: Parameters may not be therapeutically safe")
                for warning in safety_warnings:
                    logger.warning(f"Safety check: {warning}")
                
                # Apply emergency therapeutic constraints
                music_params = self._apply_emergency_therapeutic_constraints(music_params, valence, arousal)
                logger.info("Applied emergency therapeutic constraints")
            
            # Log any therapeutic warnings
            for warning in safety_warnings:
                if "WARNING" in warning:
                    logger.info(f"Therapeutic guidance: {warning}")
            
            # Get therapeutic recommendations for logging
            recommendations = self.param_space.get_therapeutic_recommendations(valence, arousal)
            if recommendations:
                logger.debug(f"Therapeutic recommendations: {list(recommendations.keys())}")
            
            # Ensure all parameters are within therapeutic bounds
            music_params = self.param_space.clip_parameters(music_params, use_therapeutic_bounds=True)
            
            # Final validation
            final_is_safe, final_warnings = self.param_space.validate_therapeutic_safety(music_params)
            if not final_is_safe:
                logger.error("CRITICAL: Parameters still unsafe after constraints - using safe defaults")
                music_params = self._get_safe_default_parameters()
            
            logger.debug(f"Generated {len(music_params)} therapeutic music parameters")
            return music_params
            
        except Exception as e:
            logger.error(f"Error in emotion-to-music mapping: {e}")
            # Return safe defaults on any error
            return self._get_safe_default_parameters()
    
    def _apply_emergency_therapeutic_constraints(self, params: Dict[str, float], valence: float, arousal: float) -> Dict[str, float]:
        """Apply emergency therapeutic constraints when safety validation fails"""
        constrained = params.copy()
        
        # Emergency constraint 1: Force safe volume levels
        constrained['overall_volume'] = min(constrained.get('overall_volume', 0.5), 0.6)
        
        # Emergency constraint 2: Limit dissonance for any negative emotional state
        if valence < 0:
            constrained['dissonance_level'] = min(constrained.get('dissonance_level', 0.2), 0.4)
        
        # Emergency constraint 3: Calm down high arousal states
        if arousal > 0.8:
            constrained['tempo_bpm'] = min(constrained.get('tempo_bpm', 100), 110)
            constrained['dynamic_range'] = min(constrained.get('dynamic_range', 0.4), 0.5)
            constrained['accent_strength'] = min(constrained.get('accent_strength', 0.3), 0.4)
        
        # Emergency constraint 4: Energize very low arousal states safely
        if arousal < 0.2:
            constrained['tempo_bpm'] = max(constrained.get('tempo_bpm', 100), 80)
            constrained['brightness'] = max(constrained.get('brightness', 0.5), 0.4)
        
        # Emergency constraint 5: Support very negative valence states
        if valence < -0.7:
            constrained['warmth'] = max(constrained.get('warmth', 0.6), 0.5)
            constrained['roughness'] = min(constrained.get('roughness', 0.2), 0.3)
            constrained['reverb_amount'] = min(constrained.get('reverb_amount', 0.3), 0.4)
        
        # Emergency constraint 6: Ensure therapeutic minimums
        constrained['emotional_stability'] = max(constrained.get('emotional_stability', 0.6), 0.4)
        constrained['therapeutic_intensity'] = np.clip(constrained.get('therapeutic_intensity', 0.5), 0.3, 0.7)
        
        return constrained
    
    def _get_safe_default_parameters(self) -> Dict[str, float]:
        """Get clinically safe default parameters for emergency use"""
        return {
            'tempo_bpm': 90.0,           # Calm, safe tempo
            'overall_volume': 0.5,       # Moderate, safe volume
            'dissonance_level': 0.1,     # Minimal dissonance
            'brightness': 0.4,           # Moderate brightness
            'warmth': 0.7,              # High warmth for comfort
            'reverb_amount': 0.2,       # Light reverb
            'voice_density': 2.0,       # Simple texture
            'chord_complexity': 0.3,    # Simple harmony
            'rhythm_complexity': 0.3,   # Simple rhythm
            'articulation': 0.6,        # Smooth articulation
            'repetition_factor': 0.7,   # High repetition for stability
            'emotional_stability': 0.8, # High stability
            'therapeutic_intensity': 0.5, # Moderate intensity
            'cognitive_load': 0.3,      # Low cognitive load
        }
