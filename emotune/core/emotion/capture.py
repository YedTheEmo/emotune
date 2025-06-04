import os
import time
import threading
import queue
from datetime import datetime
import cv2
import sounddevice as sd
import numpy as np
from typing import Dict, Optional, Tuple

from utils.logging import get_logger
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
        
        # Camera setup
        self.camera = None
        self.last_face_time = 0
        
    def start_capture(self):
        """Start continuous capture in background thread"""
        if self.running:
            return
            
        self.running = True
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise RuntimeError("Failed to open camera")
            
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        logger.info("Emotion capture started")
        
    def stop_capture(self):
        """Stop capture and cleanup"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        if self.camera:
            self.camera.release()
        logger.info("Emotion capture stopped")
        
    def start(self):
        self.start_capture()
        return True  # to keep the previous contract if needed

    def stop(self):
        self.stop_capture()

    def get_data(self):
        return self.get_latest_data()

    def get_latest_data(self) -> Optional[Dict]:
        """Get latest captured data (non-blocking)"""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None
            
    def _capture_loop(self):
        """Main capture loop running in background thread (non-blocking audio capture)"""
        import collections
        audio_buffer = collections.deque()
        audio_frames_needed = int(self.audio_duration * self.sample_rate)
        audio_accum = []
        last_audio_time = time.time()

        def audio_callback(indata, frames, time_info, status):
            audio_buffer.append(indata.copy())

        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype=np.float32, callback=audio_callback):
            while self.running:
                current_time = time.time()
                data = {'timestamp': current_time}

                # Face capture at specified FPS
                if current_time - self.last_face_time >= (1.0 / self.face_fps):
                    ret, frame = self.camera.read()
                    if ret:
                        data['face_frame'] = frame.copy()
                        self.last_face_time = current_time

                # Non-blocking audio chunk collection
                # Accumulate audio frames until enough for one chunk
                while audio_buffer and len(audio_accum) < audio_frames_needed:
                    audio_accum.append(audio_buffer.popleft())
                if len(audio_accum) >= audio_frames_needed:
                    audio_data = np.concatenate(audio_accum)[:audio_frames_needed]
                    data['audio_chunk'] = audio_data.flatten()
                    audio_accum = []
                    last_audio_time = current_time

                # Only add to queue if we have at least face or audio
                if 'face_frame' in data or 'audio_chunk' in data:
                    try:
                        self.data_queue.put_nowait(data)
                    except queue.Full:
                        try:
                            self.data_queue.get_nowait()
                            self.data_queue.put_nowait(data)
                        except queue.Empty:
                            pass

                time.sleep(0.01)
                
    def is_running(self) -> bool:
        """Return True if emotion capture is running."""
        return self.running
