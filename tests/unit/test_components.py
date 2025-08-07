#!/usr/bin/env python3
"""
Test individual components to isolate initialization issues
"""

import sys
import os

# Add project root to path (now in tests/unit subdirectory)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_emotion_capture():
    """Test EmotionCapture initialization"""
    print("Testing EmotionCapture...")
    try:
        from emotune.core.emotion.capture import EmotionCapture
        capture = EmotionCapture(face_fps=5, audio_duration=0.5, sample_rate=16000, camera_index=0)
        print("✓ EmotionCapture created successfully")
        return True
    except Exception as e:
        print(f"✗ EmotionCapture failed: {e}")
        return False

def test_emotion_analyzer():
    """Test EmotionAnalyzer initialization"""
    print("Testing EmotionAnalyzer...")
    try:
        from emotune.core.emotion.analyzers import EmotionAnalyzer
        analyzer = EmotionAnalyzer(device='cpu')
        print("✓ EmotionAnalyzer created successfully")
        return True
    except Exception as e:
        print(f"✗ EmotionAnalyzer failed: {e}")
        return False

def test_music_engines():
    """Test music engine initialization"""
    print("Testing music engines...")
    try:
        from emotune.core.music.engine_pyo import create_music_engine
        from emotune.core.music.engine_tidal import TidalEngine
        from emotune.core.music.engine_sonicpi import SonicPiEngine
        from emotune.core.music.engine_midi import MidiEngine
        
        # Test each engine
        pyo_engine = create_music_engine("auto")
        print("✓ Pyo engine created")
        
        tidal_engine = TidalEngine()
        print("✓ Tidal engine created")
        
        sonicpi_engine = SonicPiEngine()
        print("✓ SonicPi engine created")
        
        midi_engine = MidiEngine()
        print("✓ Midi engine created")
        
        return True
    except Exception as e:
        print(f"✗ Music engines failed: {e}")
        return False

def test_core_components():
    """Test core emotion components"""
    print("Testing core components...")
    try:
        from emotune.core.emotion.emotion_state import EmotionState
        from emotune.core.emotion.kalman_filter import KalmanEmotionFilter
        from emotune.core.trajectory.planner import TrajectoryPlanner
        from emotune.core.trajectory.dtw_matcher import DTWMatcher
        from emotune.core.emotion.fusion import EmotionFusion
        from emotune.core.music.parameter_space import MusicParameterSpace
        from emotune.core.music.base_mapping import BaseMappingEngine
        from emotune.core.music.renderer import MusicRenderer
        
        # Test each component
        emotion_state = EmotionState()
        print("✓ EmotionState created")
        
        kalman_filter = KalmanEmotionFilter()
        print("✓ KalmanEmotionFilter created")
        
        trajectory_planner = TrajectoryPlanner()
        print("✓ TrajectoryPlanner created")
        
        dtw_matcher = DTWMatcher()
        print("✓ DTWMatcher created")
        
        emotion_fusion = EmotionFusion()
        print("✓ EmotionFusion created")
        
        param_space = MusicParameterSpace()
        print("✓ MusicParameterSpace created")
        
        music_mapping = BaseMappingEngine()
        print("✓ BaseMappingEngine created")
        
        music_renderer = MusicRenderer()
        print("✓ MusicRenderer created")
        
        return True
    except Exception as e:
        print(f"✗ Core components failed: {e}")
        return False

def main():
    """Run all component tests"""
    print("Component Initialization Test Suite")
    print("=" * 40)
    
    tests = [
        ("Core Components", test_core_components),
        ("Music Engines", test_music_engines),
        ("Emotion Analyzer", test_emotion_analyzer),
        ("Emotion Capture", test_emotion_capture),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("TEST SUMMARY")
    print("=" * 40)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 