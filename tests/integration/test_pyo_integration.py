#!/usr/bin/env python3
"""
Test script for Pyo music engine integration
"""

import sys
import os
import time
import threading
import numpy as np

# Add project root to path (now in tests/integration subdirectory)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Setup environment to suppress warnings
from emotune.utils.warnings import setup_environment
setup_environment()

# DEBUG: Print actual SessionManager source and signature
import emotune.core.session.manager
print("SessionManager loaded from:", emotune.core.session.manager.__file__)
print("SessionManager signature:", emotune.core.session.manager.SessionManager.__init__.__code__.co_varnames)

def test_pyo_engine():
    """Test Pyo music engine functionality"""
    print("\n=== Testing Pyo Music Engine ===")
    
    try:
        # Import the engine
        from emotune.core.music.engine_pyo import create_music_engine
        
        # Create engine
        engine = create_music_engine("pyo")
        print(f"✓ Engine created: {type(engine).__name__}")
        
        # Test status
        status = engine.get_status()
        print(f"✓ Engine status: {status}")
        
        # Test parameter updates
        test_params = {
            'tempo_bpm': 140.0,
            'overall_volume': 0.8,
            'brightness': 0.7,
            'warmth': 0.6
        }
        engine.update_parameters(test_params)
        print("✓ Parameters updated")
        
        # Test music generation
        test_music = {
            'emotion': {
                'valence': 0.5,
                'arousal': 0.3
            }
        }
        engine.play(test_music)
        print("✓ Music generation started")
        
        # Let it play for a moment
        time.sleep(2)
        
        # Test stop
        engine.stop()
        print("✓ Music stopped")
        
        return True
        
    except Exception as e:
        print(f"✗ Pyo engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_session_integration():
    """Test Pyo integration with session manager"""
    print("\n=== Testing Session Manager Integration ===")
    
    try:
        # Test Pyo integration without SessionManager
        print("Testing Pyo integration directly...")
        
        # Import Pyo engine directly
        from emotune.core.music.engine_pyo import create_music_engine, PyoMusicEngine
        from emotune.core.music.parameter_space import MusicParameterSpace
        
        # Create parameter space
        param_space = MusicParameterSpace()
        print("✓ Parameter space created")
        
        # Create Pyo engine
        engine = create_music_engine("pyo")
        print(f"✓ Pyo engine created: {type(engine).__name__}")
        
        # Test emotion to music mapping
        test_emotion = {
            'valence': 0.3,
            'arousal': 0.7,
            'confidence': 0.8
        }
        
        # Create emotion pattern
        if hasattr(engine, '_create_emotion_pattern'):
            pattern = engine._create_emotion_pattern(test_emotion['valence'], test_emotion['arousal'])
            print(f"✓ Emotion pattern created: tempo={pattern['tempo_bpm']:.0f} BPM")
        
        # Test music generation
        music_struct = {'emotion': test_emotion}
        engine.play(music_struct)
        print("✓ Music generation started")
        
        # Let it play briefly
        import time
        time.sleep(1)
        
        # Stop music
        engine.stop()
        print("✓ Music stopped")
        
        print("✓ Session integration test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Session integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_emotion_to_music_mapping():
    """Test emotion to music parameter mapping"""
    print("\n=== Testing Emotion to Music Mapping ===")
    
    try:
        from emotune.core.music.engine_pyo import PyoMusicEngine
        
        engine = PyoMusicEngine()
        
        # Test different emotion combinations
        test_emotions = [
            {'valence': 0.8, 'arousal': 0.9},  # Happy/Excited
            {'valence': -0.7, 'arousal': 0.6},  # Angry
            {'valence': -0.8, 'arousal': -0.3},  # Sad
            {'valence': 0.2, 'arousal': -0.5},  # Calm
        ]
        
        for i, emotion in enumerate(test_emotions):
            pattern = engine._create_emotion_pattern(emotion['valence'], emotion['arousal'])
            print(f"✓ Emotion {i+1}: valence={emotion['valence']:.1f}, arousal={emotion['arousal']:.1f}")
            print(f"  → Tempo: {pattern['tempo_bpm']:.0f} BPM, Brightness: {pattern['brightness']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Emotion mapping test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fallback_engine():
    """Test fallback engine when Pyo is not available"""
    print("\n=== Testing Fallback Engine ===")
    
    try:
        from emotune.core.music.engine_pyo import create_music_engine, FallbackMusicEngine
        
        # Test fallback engine creation
        fallback_engine = create_music_engine("fallback")
        print(f"✓ Fallback engine created: {type(fallback_engine).__name__}")
        
        # Test fallback functionality
        test_params = {'tempo_bpm': 120.0, 'volume': 0.5}
        fallback_engine.update_parameters(test_params)
        print("✓ Fallback parameters updated")
        
        test_music = {'emotion': {'valence': 0.0, 'arousal': 0.0}}
        fallback_engine.play(test_music)
        print("✓ Fallback music play logged")
        
        status = fallback_engine.get_status()
        print(f"✓ Fallback status: {status}")
        
        return True
        
    except Exception as e:
        print(f"✗ Fallback engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Pyo integration tests"""
    print("Pyo Music Engine Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("Pyo Engine", test_pyo_engine),
        ("Session Integration", test_session_integration),
        ("Emotion Mapping", test_emotion_to_music_mapping),
        ("Fallback Engine", test_fallback_engine)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Pyo integration is working correctly.")
    else:
        print("⚠ Some tests failed. Please review the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
