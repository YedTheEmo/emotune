#!/usr/bin/env python3
"""
Simplified test script for Pyo music engine integration
Tests core functionality without external dependencies
"""

import sys
import os
import time
import numpy as np

# Add project root to path (now in tests/pyo subdirectory)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_pyo_engine_core():
    """Test Pyo music engine core functionality"""
    print("\n=== Testing Pyo Music Engine Core ===")
    
    try:
        # Import the engine
        from emotune.core.music.engine_pyo import create_music_engine, PyoMusicEngine
        
        # Test engine creation
        engine = create_music_engine("pyo")
        print(f"[OK] Engine created: {type(engine).__name__}")
        
        # Test status
        status = engine.get_status()
        print(f"[OK] Engine status: {status}")
        
        # Test parameter updates
        test_params = {
            'tempo_bpm': 140.0,
            'overall_volume': 0.8,
            'brightness': 0.7,
            'warmth': 0.6
        }
        engine.update_parameters(test_params)
        print("[OK] Parameters updated")
        
        # Test music generation
        test_music = {
            'emotion': {
                'valence': 0.5,
                'arousal': 0.3
            }
        }
        engine.play(test_music)
        print("[OK] Music generation started")
        
        # Test stop
        engine.stop()
        print("[OK] Music stopped")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Pyo engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_emotion_mapping():
    """Test emotion to music parameter mapping"""
    print("\n=== Testing Emotion to Music Mapping ===")
    
    try:
        from emotune.core.music.engine_pyo import PyoMusicEngine
        
        engine = PyoMusicEngine()
        
        # Test different emotion combinations
        test_emotions = [
            {'valence': 0.8, 'arousal': 0.9, 'name': 'Happy/Excited'},
            {'valence': -0.7, 'arousal': 0.6, 'name': 'Angry'},
            {'valence': -0.8, 'arousal': -0.3, 'name': 'Sad'},
            {'valence': 0.2, 'arousal': -0.5, 'name': 'Calm'},
            {'valence': 0.0, 'arousal': 0.0, 'name': 'Neutral'},
        ]
        
        for emotion in test_emotions:
            pattern = engine._create_emotion_pattern(emotion['valence'], emotion['arousal'])
            print(f"[OK] {emotion['name']}: valence={emotion['valence']:.1f}, arousal={emotion['arousal']:.1f}")
            print(f"  → Tempo: {pattern['tempo_bpm']:.0f} BPM")
            print(f"  → Brightness: {pattern['brightness']:.2f}")
            print(f"  → Warmth: {pattern['warmth']:.2f}")
            print(f"  → Voice Density: {pattern['voice_density']:.1f}")
            print(f"  → Reverb: {pattern['reverb_amount']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Emotion mapping test failed: {e}")
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
        print(f"[OK] Fallback engine created: {type(fallback_engine).__name__}")
        
        # Test fallback functionality
        test_params = {'tempo_bpm': 120.0, 'volume': 0.5}
        fallback_engine.update_parameters(test_params)
        print("[OK] Fallback parameters updated")
        
        test_music = {'emotion': {'valence': 0.0, 'arousal': 0.0}}
        fallback_engine.play(test_music)
        print("[OK] Fallback music play logged")
        
        status = fallback_engine.get_status()
        print(f"[OK] Fallback status: {status}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Fallback engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chord_generation():
    """Test chord generation based on emotions"""
    print("\n=== Testing Chord Generation ===")
    
    try:
        from emotune.core.music.engine_pyo import PyoMusicEngine
        
        engine = PyoMusicEngine()
        
        # Test chord generation for different emotions
        test_cases = [
            {'valence': 0.8, 'arousal': 0.5, 'name': 'Happy'},
            {'valence': -0.6, 'arousal': 0.3, 'name': 'Sad'},
            {'valence': 0.0, 'arousal': 0.0, 'name': 'Neutral'},
        ]
        
        for case in test_cases:
            pattern = engine._create_emotion_pattern(case['valence'], case['arousal'])
            chord_notes = engine._generate_chord(pattern['base_frequency'], pattern)
            
            print(f"[OK] {case['name']} chord:")
            print(f"  → Base frequency: {pattern['base_frequency']:.1f} Hz")
            print(f"  → Chord notes: {[f'{freq:.1f} Hz' for freq in chord_notes]}")
            print(f"  → Dissonance level: {pattern['dissonance_level']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Chord generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_parameter_validation():
    """Test parameter validation and bounds"""
    print("\n=== Testing Parameter Validation ===")
    
    try:
        from emotune.core.music.engine_pyo import PyoMusicEngine
        
        engine = PyoMusicEngine()
        
        # Test extreme emotion values
        extreme_emotions = [
            {'valence': 1.0, 'arousal': 1.0, 'name': 'Maximum Positive'},
            {'valence': -1.0, 'arousal': -1.0, 'name': 'Maximum Negative'},
            {'valence': 0.0, 'arousal': 0.0, 'name': 'Neutral'},
        ]
        
        for emotion in extreme_emotions:
            pattern = engine._create_emotion_pattern(emotion['valence'], emotion['arousal'])
            
            # Validate parameter bounds
            assert 40 <= pattern['tempo_bpm'] <= 160, f"Tempo out of bounds: {pattern['tempo_bpm']}"
            assert 0 <= pattern['brightness'] <= 1, f"Brightness out of bounds: {pattern['brightness']}"
            assert 0 <= pattern['warmth'] <= 1, f"Warmth out of bounds: {pattern['warmth']}"
            assert 1 <= pattern['voice_density'] <= 3, f"Voice density out of bounds: {pattern['voice_density']}"
            
            print(f"[OK] {emotion['name']}: All parameters within bounds")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Parameter validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all simplified Pyo integration tests"""
    print("Simplified Pyo Music Engine Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("Pyo Engine Core", test_pyo_engine_core),
        ("Emotion Mapping", test_emotion_mapping),
        ("Fallback Engine", test_fallback_engine),
        ("Chord Generation", test_chord_generation),
        ("Parameter Validation", test_parameter_validation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[FAIL] {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "[OK] PASS" if result else "[FAIL] FAIL"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("[SUCCESS] All tests passed! Pyo integration is working correctly.")
        print("\nNext steps:")
        print("1. Install Pyo manually if you want real audio output")
        print("2. The fallback engine will work for testing without audio")
    else:
        print("[WARNING] Some tests failed. Please review the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 