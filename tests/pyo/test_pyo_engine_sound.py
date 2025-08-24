import os
import sys
import time

try:
    from emotune.core.music.engine_pyo import PYO_AVAILABLE, PyoMusicEngine
except Exception as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Skip in CI or when Pyo is unavailable
if os.environ.get('CI') == 'true':
    print("SKIP: CI environment")
    sys.exit(0)
if not PYO_AVAILABLE:
    print("SKIP: Pyo not available")
    sys.exit(0)

def main():
    engine = PyoMusicEngine()
    try:
        params = {
            'tempo_bpm': 90.0,
            'brightness': 0.5,
            'warmth': 0.5,
            'rhythm_complexity': 0.4,
            'dissonance_level': 0.1,
            'voice_density': 2.0,
            'reverb_amount': 0.2,
            'articulation': 0.5,
            'overall_volume': 0.5,
        }
        emotion = {'valence': 0.0, 'arousal': 0.3}
        engine.play({'parameters': params, 'emotion': emotion})
        time.sleep(0.8)
        if not engine.mixer or not engine.server:
            print("FAIL: Mixer or server not initialized")
            return 1
        # Check for pattern playing (give second chance)
        if not engine.pattern_player or not getattr(engine.pattern_player, 'isPlaying', lambda: False)():
            time.sleep(0.8)
        if not engine.pattern_player or not engine.pattern_player.isPlaying():
            print("FAIL: Pattern did not start playing")
            return 1
        # Trigger a note to validate routing; should not raise
        try:
            engine._play_note()
        except Exception as e:
            print(f"FAIL: _play_note raised: {e}")
            return 1
        print("PASS: Pyo engine started and scheduled notes")
        return 0
    finally:
        try:
            engine.stop()
        except Exception:
            pass

if __name__ == '__main__':
    sys.exit(main()) 