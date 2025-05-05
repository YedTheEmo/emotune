import os
import subprocess
import threading
import time
import random
import signal
import sys

class MusicController:
    def __init__(self):
        self.foxdot_process = None
        self.current_params = None
        self._stop_event = threading.Event()
        self.lock = threading.Lock()
        self.base_scale = [0, 2, 4, 5, 7, 9, 11]  # Major scale intervals
        self.base_pattern = [0, 2, 4, 5]

    def start_interactive_session(self):
        """Start a single long-lived interactive session."""
        if self.foxdot_process and self.foxdot_process.poll() is None:
            return  # Session is already running

        # Start a long-lived interactive session
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script_path = os.path.join(project_root, "music_engine", "live_music.py")

        self.foxdot_process = subprocess.Popen(
            ["python", "-i", script_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        )

    def map_va_to_music(self, valence, arousal):
        """Map the valence and arousal to music parameters."""
        valence = max(-1.0, min(1.0, valence))
        arousal = max(-1.0, min(1.0, arousal))

        v_norm = (valence + 1) / 2
        a_norm = (arousal + 1) / 2

        return {
            'bpm': 80 + int(a_norm * 60),
            'scale': self._select_scale(v_norm),
            'root': random.randint(0, 11),
            'synth': self._select_synth(v_norm, a_norm),
            'harmony': self._create_harmony(v_norm, a_norm),
            'rhythm': self._create_rhythm(a_norm)
        }

    def _select_scale(self, valence):
        """Select major or minor scale based on valence."""
        return "major" if valence > 0.5 else "minor"

    def _select_synth(self, v, a):
        """Select the synth type based on valence and arousal."""
        if a > 0.7:
            return "bass" if v < 0.5 else "pluck"
        elif a > 0.4:
            return "pads" if v > 0.5 else "saw"
        return "ambi"

    def _create_harmony(self, v, a):
        """Create a harmony pattern based on valence and arousal."""
        pattern = self.base_pattern.copy()
        if a > 0.7:
            pattern.append(pattern[-1] + 2)
        if v < 0.3:
            pattern = [x - 1 for x in pattern]
        return pattern

    def _create_rhythm(self, arousal):
        """Create a rhythm pattern based on arousal."""
        base_dur = [0.5, 0.25, 0.25] if arousal > 0.5 else [1, 0.5]
        return base_dur * 2

    def generate_foxdot_code(self, params):
        """Generate the FoxDot code to be sent to the interactive session."""
        return f"""
from FoxDot import *

Clock.bpm = {params['bpm']}
Scale.default = Scale.{params['scale']}
Root.default = {params['root']}

p1 >> {params['synth']}(
    {params['harmony']},
    dur={params['rhythm']},
    amp=0.7
)

d1 >> play("x-o-", sample=2, rate=1.2, amp={min(1.0, params['bpm']/150)})
"""

    def play_music(self, valence, arousal):
        """Update the music parameters and send them to the interactive session."""
        with self.lock:
            music_params = self.map_va_to_music(valence, arousal)
            self.current_params = music_params

            # Generate the new FoxDot code
            code = self.generate_foxdot_code(music_params)

            # If the FoxDot process is already running, send the updated code
            if self.foxdot_process and self.foxdot_process.poll() is None:
                try:
                    # Send the new FoxDot code to the interactive session
                    self.foxdot_process.stdin.write(code.encode())
                    self.foxdot_process.stdin.flush()
                except Exception as e:
                    print(f"Error sending code to interactive session: {e}")
            else:
                # If the interactive session isn't running, start it
                self.start_interactive_session()
                self.play_music(valence, arousal)  # Retry once the session is started

    def stop_music(self):
        """Stop the interactive session and music."""
        with self.lock:
            if self.foxdot_process and self.foxdot_process.poll() is None:
                try:
                    # Gracefully exit the interactive session
                    self.foxdot_process.stdin.write("exit()\n".encode())
                    self.foxdot_process.stdin.flush()
                except Exception as e:
                    print(f"Error stopping music: {e}")
                self.foxdot_process = None

    def start_music_loop(self, emotion_service):
        """Start the music loop, updating music every few seconds based on emotion data."""
        def update_loop():
            while not self._stop_event.is_set():
                try:
                    data = emotion_service.get_raw_data()
                    self.play_music(data['valence'], data['arousal'])
                    time.sleep(10)  # Update every 10 seconds
                except Exception as e:
                    print(f"Music loop error: {e}")
                time.sleep(5)  # Add some delay to avoid overloading the process

        self._stop_event.clear()
        threading.Thread(target=update_loop, daemon=True).start()

