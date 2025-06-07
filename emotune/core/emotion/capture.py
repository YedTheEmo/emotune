import os
import time
import threading
import queue
from datetime import datetime
import cv2
import sounddevice as sd
import numpy as np
from typing import Dict, Optional, Tuple

from emotune.utils.logging import get_logger
logger = get_logger()

class EmotionCapture:
    """Unified emotion capture from webcam and microphone"""
    
    def __init__(self, face_fps: int = 1, audio_duration: float = 3.0, 
                 sample_rate: int = 16000):
        self.face_fps = face_fps
        self.audio_duration = audio_duration
        self.sample_rate = sample_rate
        
        # Threading components
        self.capture_thread = None
        self.running = False
        self.data_queue = queue.Queue(maxsize=10)

        # Add lock to prevent race conditions
        self._start_lock = threading.Lock()
        self.camera_lock = threading.Lock()  # Add thread lock for camera access

    def start_capture(self):
        """Start continuous capture in background thread"""
        with self._start_lock:
            logger.info(f"[EmotionCapture] start_capture called. running={self.running}, capture_thread={self.capture_thread}")
            if self.running:
                logger.debug("[EmotionCapture] start_capture called but already running.")
                return
            try:
                self.running = True
                # No persistent camera initialization here
                self.capture_thread = threading.Thread(target=self._capture_loop)
                self.capture_thread.daemon = True
                self.capture_thread.start()
                logger.info("Emotion capture started successfully")
            except Exception as e:
                logger.error(f"[EmotionCapture] Failed to start capture: {e}")
                self._cleanup_on_error()
                raise

    def stop_capture(self):
        """Stop capture and cleanup"""
        with self._start_lock:
            logger.info(f"[EmotionCapture] stop_capture called. running={self.running}, capture_thread={self.capture_thread}")
            self.running = False
            if self.capture_thread:
                self.capture_thread.join(timeout=3.0)
                logger.debug("[EmotionCapture] capture_thread joined.")
            self.capture_thread = None
            logger.info("Emotion capture stopped")

    def _cleanup_on_error(self):
        """Clean up resources when start_capture fails"""
        logger.info("[EmotionCapture] Cleaning up after error...")
        self.running = False
        self.capture_thread = None
        logger.info("[EmotionCapture] Cleanup complete.")

    def get_data(self):
        return self.get_latest_data()

    def get_latest_data(self) -> Optional[Dict]:
        """Get latest captured data (blocking, with timeout for efficiency)"""
        try:
            # Wait up to 0.25s for new data (matches 5Hz emission rate)
            data = self.data_queue.get(timeout=0.25)
            logger.debug(f"[EmotionCapture] get_latest_data returned data with keys: {list(data.keys())}")
            return data
        except queue.Empty:
            # Only log at debug level to reduce log noise
            logger.debug("[EmotionCapture] get_latest_data: data_queue is empty (timeout).")
            return None
            
    def _capture_loop(self):
        """Main capture loop: open camera once, keep open, proper warmup, fixed audio chunk."""
        snapshot_interval = 0.2  # 5Hz
        last_snapshot_time = time.time()
        logger.info(f"[EmotionCapture] _capture_loop started. running={self.running}, snapshot_interval={snapshot_interval}s")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            logger.error("[EmotionCapture] Camera failed to open")
            return
        logger.info("[EmotionCapture] Camera opened successfully in _capture_loop.")
        # Proper warmup: 5 frames, 0.1s each
        for i in range(5):
            ret, warmup_frame = cap.read()
            logger.debug(f"[EmotionCapture] Warmup read {i+1}/5: ret={ret}, mean={np.mean(warmup_frame) if ret and warmup_frame is not None else 'N/A'}")
            time.sleep(0.1)
        logger.info("[EmotionCapture] Warmup complete. Entering main loop.")
        while self.running:
            now = time.time()
            if now - last_snapshot_time >= snapshot_interval:
                ret, frame = cap.read()
                logger.debug(f"[EmotionCapture] Frame read: ret={ret}, type={type(frame)}, shape={getattr(frame, 'shape', None)}")
                if not ret or frame is None:
                    logger.warning("[EmotionCapture] Invalid frame captured (ret=False or None)")
                    continue
                if not isinstance(frame, np.ndarray):
                    logger.warning(f"[EmotionCapture] Invalid frame type: {type(frame)}. Skipping frame.")
                    continue
                if frame.ndim != 3 or frame.shape[2] != 3:
                    logger.warning(f"[EmotionCapture] Unexpected frame shape: {getattr(frame, 'shape', None)}. Skipping frame.")
                    continue
                frame_mean = np.mean(frame)
                logger.debug(f"[EmotionCapture] Frame mean: {frame_mean}")
                if frame_mean < 5:  # Lower threshold
                    logger.warning(f"[EmotionCapture] Black frame detected! Mean: {frame_mean}")
                    continue
                # Convert BGR to RGB for model compatibility
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                logger.debug(f"[EmotionCapture] Frame converted from BGR to RGB.")
                # --- Save debug image in BGR for visual inspection (matches test script) ---
                try:
                    debug_dir = os.path.join(os.path.dirname(__file__), '../../../face_debug')
                    os.makedirs(debug_dir, exist_ok=True)
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    debug_path = os.path.join(debug_dir, f'frame_{ts}.jpg')
                    cv2.imwrite(debug_path, frame)  # Save original BGR frame
                    logger.debug(f"[EmotionCapture] Debug frame saved: {debug_path}")
                except Exception as e:
                    logger.error(f"[EmotionCapture] Failed to save debug frame: {e}")
                logger.debug(f"[EmotionCapture] Frame valid, preparing data for queue.")
                # Real audio capture: 0.5s chunk
                try:
                    audio_data = sd.rec(int(0.5 * self.sample_rate), samplerate=self.sample_rate, channels=1, dtype='float32')
                    sd.wait()
                    audio_data = audio_data.flatten()
                    logger.debug(f"[EmotionCapture] Audio captured: shape={audio_data.shape}, dtype={audio_data.dtype}")
                except Exception as e:
                    logger.error(f"[EmotionCapture] Audio capture failed: {e}. Using zeros.")
                    audio_data = np.zeros(int(0.5 * self.sample_rate), dtype=np.float32)
                data = {
                    'timestamp': now,
                    'face_frame': frame_rgb.copy(),
                    'audio_chunk': audio_data,
                    'sr': self.sample_rate
                }
                try:
                    self.data_queue.put_nowait(data)
                    logger.debug(f"[EmotionCapture] Data put in queue. Keys: {list(data.keys())}")
                except queue.Full:
                    logger.warning("[EmotionCapture] Data queue full, dropping oldest.")
                    try:
                        self.data_queue.get_nowait()
                        self.data_queue.put_nowait(data)
                        logger.debug("[EmotionCapture] Oldest data dropped, new data put in queue.")
                    except queue.Empty:
                        logger.error("[EmotionCapture] Data queue full and empty on get_nowait().")
                logger.log_emotion(
                    valence=np.mean(frame_rgb) if frame_rgb is not None else 0,
                    arousal=0,  # Placeholder, real arousal should be computed if available
                    confidence=1.0 if frame_rgb is not None else 0.0,
                    source='face',
                    raw_data={'frame_shape': frame_rgb.shape if frame_rgb is not None else None}
                )
                last_snapshot_time = now
            time.sleep(0.01)
        logger.info("[EmotionCapture] Exiting _capture_loop, releasing camera.")
        cap.release()

    def is_running(self) -> bool:
        """Return True if emotion capture is running and thread is alive."""
        return self.running and self.capture_thread is not None and self.capture_thread.is_alive()

    def start(self):
        self.start_capture()
        return True

    def stop(self):
        self.stop_capture()
