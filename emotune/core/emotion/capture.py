import os
import time
import threading
import queue
from datetime import datetime
import cv2
import sounddevice as sd
import numpy as np
from typing import Dict, Optional, Tuple
import weakref

from emotune.utils.logging import get_logger
logger = get_logger()

class EmotionCapture:
    """Unified emotion capture from webcam and microphone with proper resource management"""
    
    def __init__(self, face_fps: int = 5, audio_duration: float = 0.5, 
                 sample_rate: int = 16000, camera_index: int = 0):
        self.face_fps = face_fps
        self.audio_duration = audio_duration
        self.sample_rate = sample_rate
        self.camera_index = camera_index
        
        # Threading components
        self.capture_thread = None
        self.running = False
        self.data_queue = queue.Queue(maxsize=20)  # Increased buffer size
        
        # Resource management
        self._start_lock = threading.Lock()
        self._camera_lock = threading.Lock()
        self._camera = None
        self._camera_initialized = False
        
        # ENHANCED CLEANUP: Audio device tracking
        self._audio_device_info = None
        self._audio_stream = None
        self._cleanup_attempted = False
        
        # Performance tracking
        self._last_frame_time = 0
        self._frame_interval = 1.0 / face_fps
        self._capture_stats = {
            'frames_captured': 0,
            'frames_dropped': 0,
            'audio_chunks': 0,
            'errors': 0
        }
        # Last frame for preview streaming
        self._last_frame = None
        self._last_frame_lock = threading.Lock()
        
        # Audio configuration
        self._audio_device = None
        self._audio_buffer_size = int(audio_duration * sample_rate)
        # NEW: Keep the latest audio chunk to combine with frames
        self._latest_audio = None
        self._audio_lock = threading.Lock()
        
        # ENHANCED CLEANUP: Shutdown event for thread coordination
        self._shutdown_event = threading.Event()
        self._force_stop_event = threading.Event()
        
        # Register cleanup on object destruction
        weakref.finalize(self, self._cleanup_resources)

    def _cleanup_resources(self):
        """Enhanced cleanup resources when object is destroyed"""
        if self._cleanup_attempted:
            return
        self._cleanup_attempted = True
        
        try:
            logger.info("Performing final resource cleanup...")
            
            # Stop capture if still running
            if self.running:
                self.stop_capture()
            
            # Force cleanup audio resources
            self._force_cleanup_audio()
            
            # Force cleanup camera resources
            self._force_cleanup_camera()
            
        except Exception as e:
            logger.error(f"Error during final cleanup: {e}")
        finally:
            with self._last_frame_lock:
                self._last_frame = None

    def _initialize_camera(self) -> bool:
        """Initialize camera with proper error handling and retry logic"""
        with self._camera_lock:
            if self._camera_initialized and self._camera is not None:
                return True
                
            try:
                # Try different camera backends
                backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
                for backend in backends:
                    try:
                        self._camera = cv2.VideoCapture(self.camera_index, backend)
                        if self._camera.isOpened():
                            # Configure camera properties for better performance
                            self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            self._camera.set(cv2.CAP_PROP_FPS, 30)
                            self._camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
                            
                            # Warm up camera
                            for _ in range(3):
                                ret, _ = self._camera.read()
                                if not ret:
                                    break
                                time.sleep(0.1)
                            
                            if ret:
                                self._camera_initialized = True
                                logger.info(f"Camera initialized successfully with backend {backend}")
                                return True
                            else:
                                self._camera.release()
                                self._camera = None
                    except Exception as e:
                        logger.warning(f"Failed to initialize camera with backend {backend}: {e}")
                        if self._camera:
                            self._camera.release()
                            self._camera = None
                
                logger.error("Failed to initialize camera with any backend")
                return False
                
            except Exception as e:
                logger.error(f"Camera initialization error: {e}")
                return False

    def _release_camera(self):
        """Safely release camera resources"""
        with self._camera_lock:
            if self._camera is not None:
                try:
                    self._camera.release()
                except Exception as e:
                    logger.error(f"Error releasing camera: {e}")
                finally:
                    self._camera = None
                    self._camera_initialized = False

    def start_capture(self):
        """Start continuous capture in background thread with proper initialization"""
        with self._start_lock:
            if self.running:
                logger.debug("Capture already running")
                return
                
            try:
                # Initialize camera first
                if not self._initialize_camera():
                    raise RuntimeError("Failed to initialize camera")
                
                self.running = True
                self.capture_thread = threading.Thread(
                    target=self._capture_loop, 
                    name="EmotionCapture",
                    daemon=True
                )
                self.capture_thread.start()
                logger.info("Emotion capture started successfully")
                
            except Exception as e:
                logger.error(f"Failed to start capture: {e}")
                self._cleanup_on_error()
                raise

    def stop_capture(self):
        """Enhanced stop capture with comprehensive cleanup and forced termination"""
        with self._start_lock:
            if not self.running:
                return
                
            logger.info("Stopping emotion capture with enhanced cleanup...")
            
            # Phase 1: Signal shutdown
            self.running = False
            self._shutdown_event.set()
            
            # Phase 2: Wait for graceful thread termination
            if self.capture_thread is not None and hasattr(self.capture_thread, 'is_alive') and self.capture_thread.is_alive():
                logger.info("Waiting for capture thread to terminate gracefully...")
                self.capture_thread.join(timeout=2.0)
                
                # Phase 3: Force termination if needed
                if self.capture_thread is not None and hasattr(self.capture_thread, 'is_alive') and self.capture_thread.is_alive():
                    logger.warning("Capture thread did not exit gracefully, forcing shutdown...")
                    self._force_stop_event.set()
                    self.capture_thread.join(timeout=1.0)
                    
                    if self.capture_thread is not None and hasattr(self.capture_thread, 'is_alive') and self.capture_thread.is_alive():
                        logger.error("Capture thread did not respond to forced shutdown")
                        # Note: Python doesn't support thread.terminate(), but we can abandon it
            
            # Phase 4: Resource cleanup
            self._enhanced_resource_cleanup()
            
            # Phase 5: Reset state
            self.capture_thread = None
            self._shutdown_event.clear()
            self._force_stop_event.clear()
            
            logger.info("Emotion capture stopped with enhanced cleanup")

    def _cleanup_on_error(self):
        """Clean up resources when start_capture fails"""
        logger.info("Cleaning up after error...")
        self.running = False
        self.capture_thread = None
        self._release_camera()

    def get_data(self) -> Optional[Dict]:
        """Get latest captured data with non-blocking timeout"""
        return self.get_latest_data()

    def get_latest_data(self) -> Optional[Dict]:
        """Get latest captured data with efficient timeout handling"""
        try:
            # Use shorter timeout for better responsiveness
            data = self.data_queue.get(timeout=0.1)
            return data
        except queue.Empty:
            return None

    def _enhanced_resource_cleanup(self):
        """Comprehensive resource cleanup for camera and audio"""
        logger.info("Performing enhanced resource cleanup...")
        
        # Cleanup audio resources
        self._force_cleanup_audio()
        
        # Cleanup camera resources
        self._force_cleanup_camera()
        
        # Clear queue
        self._clear_data_queue()
        
        # Reset performance stats
        self._capture_stats = {
            'frames_captured': 0,
            'frames_dropped': 0,
            'audio_chunks': 0,
            'errors': 0
        }

    def _force_cleanup_audio(self):
        """Force cleanup of all audio resources"""
        try:
            # Stop any active audio streams
            if hasattr(sd, '_streams') and sd._streams:
                logger.info("Stopping active sounddevice streams...")
                for stream in list(sd._streams):
                    try:
                        if hasattr(stream, 'stop'):
                            stream.stop()
                        if hasattr(stream, 'close'):
                            stream.close()
                    except Exception as e:
                        logger.debug(f"Error stopping stream: {e}")
            
            # Force sounddevice cleanup
            try:
                sd.stop()  # Stop all streams
                sd.close()  # Close sounddevice
            except Exception as e:
                logger.debug(f"Sounddevice cleanup error: {e}")
            
            # Reset audio device info
            self._audio_device_info = None
            self._audio_stream = None
            self._audio_device = None
            
            logger.info("Audio resources forcefully cleaned up")
            
        except Exception as e:
            logger.error(f"Error during audio cleanup: {e}")

    def _force_cleanup_camera(self):
        """Force cleanup of camera resources"""
        try:
            with self._camera_lock:
                if self._camera is not None:
                    logger.info("Forcefully releasing camera...")
                    try:
                        self._camera.release()
                    except Exception as e:
                        logger.debug(f"Camera release error: {e}")
                    finally:
                        self._camera = None
                        self._camera_initialized = False
                
                # Additional OpenCV cleanup
                try:
                    cv2.destroyAllWindows()  # Close any OpenCV windows
                except Exception as e:
                    logger.debug(f"OpenCV cleanup error: {e}")
                    
            logger.info("Camera resources forcefully cleaned up")
            
        except Exception as e:
            logger.error(f"Error during camera cleanup: {e}")
        finally:
            with self._last_frame_lock:
                self._last_frame = None

    def _clear_data_queue(self):
        """Clear all data from the queue"""
        cleared_count = 0
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
                cleared_count += 1
            except queue.Empty:
                break
        
        if cleared_count > 0:
            logger.info(f"Cleared {cleared_count} items from data queue")

    def _capture_loop(self):
        """Enhanced capture loop with proper shutdown handling and resource management"""
        logger.info("Starting enhanced capture loop...")
        
        # Start a separate thread for audio capture
        audio_thread = threading.Thread(target=self._audio_capture_loop, name="AudioCapture", daemon=True)
        audio_thread.start()
        
        consecutive_frame_failures = 0
        reinit_threshold = 15
        
        try:
            while self.running and not self._shutdown_event.is_set():
                if self._force_stop_event.is_set():
                    logger.info("Force stop event detected, breaking capture loop")
                    break
                
                current_time = time.time()
                
                # Prioritize frame capture at the specified FPS
                if current_time - self._last_frame_time >= self._frame_interval:
                    frame = self._capture_frame()
                    
                    if frame is not None:
                        consecutive_frame_failures = 0
                        # Update last frame for preview
                        try:
                            with self._last_frame_lock:
                                self._last_frame = frame.copy()
                        except Exception:
                            pass
                        # Put combined data (frame + latest audio) in the queue
                        with self._audio_lock:
                            latest_audio = None if self._latest_audio is None else self._latest_audio.copy()
                        data = {
                            'timestamp': current_time,
                            'frame': frame,
                            'audio': latest_audio,
                            'sr': self.sample_rate
                        }
                        
                        try:
                            self.data_queue.put_nowait(data)
                            self._last_frame_time = current_time
                            self._capture_stats['frames_captured'] += 1
                        except queue.Full:
                            # Drop oldest data if queue is full (prefer keeping latest)
                            try:
                                self.data_queue.get_nowait()
                                self.data_queue.put_nowait(data)
                                self._capture_stats['frames_dropped'] += 1
                            except queue.Empty:
                                pass
                    else:
                        consecutive_frame_failures += 1
                        if consecutive_frame_failures >= reinit_threshold:
                            logger.warning("Repeated frame capture failures; attempting camera reinitialization")
                            self._release_camera()
                            # brief pause then try to reinit
                            time.sleep(0.2)
                            if not self._initialize_camera():
                                logger.error("Camera reinitialization failed; will retry")
                            else:
                                logger.info("Camera reinitialized successfully")
                            consecutive_frame_failures = 0
                
                # Brief sleep to yield to other threads
                time.sleep(0.001)
                    
        except Exception as e:
            logger.error(f"Fatal error in capture loop: {e}")
        finally:
            logger.info("Capture loop ended, performing cleanup...")
            self._cleanup_on_error()

    def _audio_capture_loop(self):
        """Dedicated loop for continuous audio capture"""
        logger.info("Starting audio capture loop...")
        
        try:
            while self.running and not self._shutdown_event.is_set():
                audio_chunk = self._capture_audio_safe()
                
                if audio_chunk is not None:
                    # Update latest buffered audio for pairing
                    with self._audio_lock:
                        self._latest_audio = audio_chunk
                    # Also enqueue an audio-only item to ensure voice analysis proceeds even if frames drop
                    try:
                        self.data_queue.put_nowait({
                            'timestamp': time.time(),
                            'frame': None,
                            'audio': audio_chunk,
                            'sr': self.sample_rate
                        })
                    except queue.Full:
                        pass
                    self._capture_stats['audio_chunks'] += 1
                
                # Sleep for the duration of the audio chunk to avoid constant recapture
                time.sleep(self.audio_duration)
                
        except Exception as e:
            logger.error(f"Fatal error in audio capture loop: {e}")
        finally:
            logger.info("Audio capture loop ended.")

    def _capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame with proper error handling"""
        if not self._camera_initialized or self._camera is None:
            return None
            
        try:
            with self._camera_lock:
                ret, frame = self._camera.read()
                
            if not ret or frame is None:
                return None
                
            # Validate frame
            if not isinstance(frame, np.ndarray):
                return None
                
            if frame.ndim != 3 or frame.shape[2] != 3:
                return None
                
            # Check for black frames (relaxed) and log
            frame_mean = np.mean(frame)
            frame_max = np.max(frame)
            # Normalize mean/std to 0-1 range regardless of dtype/scale
            norm_mean = float(frame_mean / 255.0) if frame_max > 1.5 else float(frame_mean)
            norm_std = float((np.std(frame) / 255.0) if frame_max > 1.5 else np.std(frame))
            # Drop only if nearly all-black and very low variance
            if norm_mean < 0.02 and norm_std < 0.01:
                logger.debug(f"[Capture] Dropping near-black frame: mean={norm_mean:.3f}, std={norm_std:.3f}")
                return None
            else:
                logger.debug(f"[Capture] Captured frame: shape={frame.shape}, mean_norm={norm_mean:.3f}, std_norm={norm_std:.3f}")
            
            # Save debug frame occasionally
            if self._capture_stats['frames_captured'] % 30 == 0:  # Every 30 frames
                self._save_debug_frame(frame)
                
            # Return raw BGR frame; analyzer will handle color conversion
            return frame
            
        except Exception as e:
            logger.error(f"Frame capture error: {e}")
            return None

    def _capture_audio_safe(self) -> Optional[np.ndarray]:
        """Simplified and robust audio capture."""
        try:
            if not self.running or self._shutdown_event.is_set():
                return None

            audio_data = sd.rec(
                self._audio_buffer_size,
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                blocking=True  # Use blocking call for simplicity
            )
            sd.wait() # Wait for recording to complete

            if audio_data is not None:
                return audio_data.flatten()
            return None

        except Exception as e:
            logger.error(f"Audio capture error: {e}")
            self._capture_stats['errors'] += 1
            return None

    def _save_debug_frame(self, frame: np.ndarray):
        """Save debug frame for troubleshooting"""
        try:
            debug_dir = os.path.join(os.path.dirname(__file__), '../../../face_debug')
            os.makedirs(debug_dir, exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            debug_path = os.path.join(debug_dir, f'frame_{ts}.jpg')
            cv2.imwrite(debug_path, frame)
        except Exception as e:
            logger.error(f"Failed to save debug frame: {e}")

    def is_running(self) -> bool:
        """Check if capture is running and thread is alive"""
        return (self.running and 
                self.capture_thread is not None and 
                hasattr(self.capture_thread, 'is_alive') and
                self.capture_thread.is_alive())

    def get_stats(self) -> Dict:
        """Get capture statistics"""
        return self._capture_stats.copy()

    def get_last_frame(self) -> Optional[np.ndarray]:
        """Get the last captured BGR frame for preview streaming."""
        with self._last_frame_lock:
            if self._last_frame is None:
                return None
            try:
                return self._last_frame.copy()
            except Exception:
                return None

    def start(self):
        """Compatibility method"""
        self.start_capture()
        return True

    def stop(self):
        """Compatibility method"""
        self.stop_capture()
