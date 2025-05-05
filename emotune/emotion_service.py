import threading
import time
from .analysis.face_analysis import FaceEmotion
from .analysis.voice_analysis import VoiceEmotion


class EmotionService:
    def __init__(self):
        self.face = FaceEmotion()
        self.voice = VoiceEmotion()

        # Initialize with empty dicts instead of None
        self.face_vad = {}
        self.voice_vad = {}

        self.current_valence = 0.0
        self.current_arousal = 0.0
        self.current_emotion = "Neutral"
        self.active_sources = ['face', 'voice']

        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

      
    def get_raw_data(self):
        with self.lock:
            # Debugging logs to check the state of the data
            print("Face VAD Data:", self.face_vad)
            print("Voice VAD Data:", self.voice_vad)
            
            # Ensure consistent data structure with fallback
            raw_data = {
                "face": (self.face_vad or {}).copy(),
                "voice": (self.voice_vad or {}).copy(),
                "valence": float(self.current_valence) if isinstance(self.current_valence, (int, float)) else 0.0,
                "arousal": float(self.current_arousal) if isinstance(self.current_arousal, (int, float)) else 0.0,
                "emotion": str(self.current_emotion or ""),
                "active_sources": list(self.active_sources) if hasattr(self.active_sources, '__iter__') else []
            }
            
            # Log the final raw data
            print("Raw Data:", raw_data)
            
            return raw_data


    def set_active_sources(self, sources):
        with self.lock:
            self.active_sources = sources

    def start_background_analysis(self):
        with self.lock:
            if self._thread and self._thread.is_alive():
                print("Emotion analysis thread is already running.")
                return

            self._stop_event.clear()
            self._thread = threading.Thread(target=self._analyze_emotion_loop, daemon=True)
            self._thread.start()
            print("Started emotion analysis thread.")

    def stop_analysis(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join()
            print("Stopped emotion analysis thread.")

    def _analyze_emotion_loop(self):
        while not self._stop_event.is_set():
            try:
                self._analyze_emotion()
            except Exception as e:
                print(f"Unexpected error during emotion analysis: {e}")
            time.sleep(1)

   
    def _analyze_emotion(self):
        with self.lock:
            current_sources = list(self.active_sources)

        face_emotions = {}
        if "face" in current_sources:
            try:
                print("Starting face capture and analysis...")
                image_path = self.face.capture_and_save()
                face_emotions = self.face.analyze_face(image_path) or {}
                print(f"Face VAD - Valence: {face_emotions.get('valence', 0):.2f}, "
                      f"Arousal: {face_emotions.get('arousal', 0):.2f}")
            except Exception as e:
                print(f"Face analysis error: {e}")
                face_emotions = {}

        voice_emotions = {}
        if "voice" in current_sources:
            try:
                print("Starting voice recording and analysis...")
                file_path = self.voice.record_and_save(duration=5)
                voice_emotions = self.voice.analyze_voice(file_path) or {}
                print(f"Voice VAD - Valence: {voice_emotions.get('valence', 0):.2f}, "
                      f"Arousal: {voice_emotions.get('arousal', 0):.2f}")
            except Exception as e:
                print(f"Voice analysis error: {e}")
                voice_emotions = {}

        with self.lock:
            # Update with valid dicts only
            self.face_vad = face_emotions
            self.voice_vad = voice_emotions

            # Calculate averages
            valence_sum = 0.0
            arousal_sum = 0.0
            count = 0

            if "face" in self.active_sources and face_emotions:
                valence_sum += float(face_emotions.get('valence', 0))
                arousal_sum += float(face_emotions.get('arousal', 0))
                count += 1

            if "voice" in self.active_sources and voice_emotions:
                valence_sum += float(voice_emotions.get('valence', 0))
                arousal_sum += float(voice_emotions.get('arousal', 0))
                count += 1

            if count > 0:
                self.current_valence = valence_sum / count
                self.current_arousal = arousal_sum / count
                self.current_emotion = self._interpret_emotion(
                    self.current_valence, self.current_arousal
                )
            else:
                self.current_valence = 0.0
                self.current_arousal = 0.0
                self.current_emotion = "Neutral"

    def get_mood(self):
        with self.lock:
            return self.current_emotion

    def set_mood(self, mood):
        with self.lock:
            self.current_emotion = mood

    def _interpret_emotion(self, v, a):
        high = 0.6
        low = 0.4
        if v > high and a > high:
            return "Excited"
        elif v > high and a < low:
            return "Calm"
        elif v < low and a > high:
            return "Angry or Fearful"
        elif v < low and a < low:
            return "Sad"
        elif low <= v <= high and a > high:
            return "Alert or Anxious"
        elif low <= v <= high and a < low:
            return "Tired or Bored"
        else:
            return "Neutral or Mixed"

