#!/usr/bin/env python3
"""
Core fixes test script for EmoTune system
Tests the core algorithmic fixes without external dependencies
"""

import sys
import os
import time
import numpy as np
import time
from unittest.mock import Mock, patch

# Add project root to path (now in tests/unit subdirectory)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_emotion_fusion():
    """Test the improved emotion fusion algorithm"""
    print("\n=== Testing Emotion Fusion ===")
    
    try:
        # Import fusion module
        from emotune.core.emotion.fusion import EmotionFusion
        
        # Initialize fusion
        fusion = EmotionFusion(face_weight=0.7, voice_weight=0.3)
        
        # Test with face data only
        face_data = {
            'emotions': {'valence': 0.5, 'arousal': 0.3},
            'confidence': 0.8
        }
        
        result = fusion.fuse_emotions(face_data=face_data, voice_data=None)
        print(f"✓ Face-only fusion: {result}")
        
        # Test with voice data only
        voice_data = {
            'emotions': {'valence': -0.2, 'arousal': 0.6},
            'confidence': 0.6
        }
        
        result = fusion.fuse_emotions(face_data=None, voice_data=voice_data)
        print(f"✓ Voice-only fusion: {result}")
        
        # Test with both modalities
        result = fusion.fuse_emotions(face_data=face_data, voice_data=voice_data)
        print(f"✓ Multi-modal fusion: {result}")
        
        # Test quality metrics
        metrics = fusion.get_fusion_quality_metrics()
        print(f"✓ Fusion quality metrics: {metrics}")
        
        return True
        
    except Exception as e:
        print(f"✗ Emotion fusion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_emotion_state():
    """Test the improved emotion state management"""
    print("\n=== Testing Emotion State ===")
    
    try:
        # Import emotion state module
        from emotune.core.emotion.emotion_state import EmotionState
        
        # Initialize emotion state
        emotion_state = EmotionState()
        
        # Test with numpy array covariance
        emotion_data = {
            'mean': {'valence': 0.3, 'arousal': -0.1},
            'covariance': np.array([[0.5, 0.1], [0.1, 0.5]]),
            'uncertainty_trace': 0.3,
            'timestamp': time.time()
        }
        
        emotion_state.update_emotion(emotion_data)
        print("✓ Emotion state updated with numpy covariance")
        
        # Test with list covariance
        emotion_data2 = {
            'mean': {'valence': 0.1, 'arousal': 0.2},
            'covariance': [[0.4, 0.0], [0.0, 0.4]],
            'uncertainty_trace': 0.4,
            'timestamp': time.time()
        }
        
        emotion_state.update_emotion(emotion_data2)
        print("✓ Emotion state updated with list covariance")
        
        # Test trajectory
        trajectory = emotion_state.get_emotion_trajectory(60.0)
        print(f"✓ Emotion trajectory: {len(trajectory)} points")
        
        # Test statistics
        stats = emotion_state.get_emotion_statistics(60.0)
        print(f"✓ Emotion statistics: {stats}")
        
        # Test latest values
        latest = emotion_state.get_latest_emotion_values()
        print(f"✓ Latest emotion values: {latest}")
        
        return True
        
    except Exception as e:
        print(f"✗ Emotion state test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_kalman_filter():
    """Test the Kalman filter integration"""
    print("\n=== Testing Kalman Filter ===")
    
    try:
        # Import Kalman filter module
        from emotune.core.emotion.kalman_filter import KalmanEmotionFilter
        
        # Initialize Kalman filter
        kalman = KalmanEmotionFilter()
        
        # Test with multiple observations
        observations = [
            {'valence': 0.5, 'arousal': 0.3, 'confidence': 0.8, 'uncertainty': 0.2},
            {'valence': 0.4, 'arousal': 0.4, 'confidence': 0.7, 'uncertainty': 0.3},
            {'valence': 0.6, 'arousal': 0.2, 'confidence': 0.9, 'uncertainty': 0.1}
        ]
        
        for i, obs in enumerate(observations):
            state, cov = kalman.update(obs)
            print(f"✓ Observation {i+1}: state={state[:2]}, cov_trace={np.trace(cov[:2,:2]):.3f}")
        
        # Test distribution
        dist = kalman.get_emotion_distribution()
        print(f"✓ Final distribution: {dist}")
        
        return True
        
    except Exception as e:
        print(f"✗ Kalman filter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_serialization():
    """Test data serialization fixes"""
    print("\n=== Testing Data Serialization ===")
    
    try:
        # Test numpy type serialization
        test_data = {
            'valence': np.float64(0.5),
            'arousal': np.float32(-0.3),
            'confidence': 0.8,
            'covariance': np.array([[0.5, 0.1], [0.1, 0.5]]),
            'parameters': {
                'tempo_bpm': np.float64(120.0),
                'volume': np.int32(75)
            }
        }
        
        # Test serialization (similar to what SessionManager does)
        def serialize_data(data):
            if isinstance(data, dict):
                return {k: serialize_data(v) for k, v in data.items()}
            elif isinstance(data, (list, tuple)):
                return [serialize_data(item) for item in data]
            elif isinstance(data, (np.integer, np.floating)):
                return float(data)
            elif isinstance(data, np.ndarray):
                return data.tolist()
            else:
                return data
        
        serialized = serialize_data(test_data)
        print(f"✓ Data serialized: {serialized}")
        
        # Verify all numpy types are converted
        def check_numpy_types(obj):
            if isinstance(obj, dict):
                return any(check_numpy_types(v) for v in obj.values())
            elif isinstance(obj, (list, tuple)):
                return any(check_numpy_types(item) for item in obj)
            else:
                return isinstance(obj, (np.integer, np.floating, np.ndarray))
        
        has_numpy = check_numpy_types(serialized)
        if not has_numpy:
            print("✓ All numpy types properly converted")
        else:
            print("✗ Numpy types still present in serialized data")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Data serialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance():
    """Test system performance"""
    print("\n=== Testing Performance ===")
    
    try:
        # Import modules
        from emotune.core.emotion.fusion import EmotionFusion
        from emotune.core.emotion.emotion_state import EmotionState
        
        # Test emotion fusion performance
        fusion = EmotionFusion()
        
        start_time = time.time()
        for i in range(100):
            face_data = {
                'emotions': {'valence': np.random.uniform(-1, 1), 'arousal': np.random.uniform(-1, 1)},
                'confidence': np.random.uniform(0.1, 1.0)
            }
            voice_data = {
                'emotions': {'valence': np.random.uniform(-1, 1), 'arousal': np.random.uniform(-1, 1)},
                'confidence': np.random.uniform(0.1, 1.0)
            }
            fusion.fuse_emotions(face_data, voice_data)
        
        fusion_time = time.time() - start_time
        print(f"✓ Emotion fusion: {fusion_time:.3f}s for 100 iterations ({100/fusion_time:.1f} ops/sec)")
        
        # Test emotion state performance
        emotion_state = EmotionState()
        
        start_time = time.time()
        for i in range(100):
            emotion_data = {
                'mean': {'valence': np.random.uniform(-1, 1), 'arousal': np.random.uniform(-1, 1)},
                'covariance': np.random.rand(2, 2),
                'uncertainty_trace': np.random.uniform(0, 1),
                'timestamp': time.time()
            }
            emotion_state.update_emotion(emotion_data)
        
        state_time = time.time() - start_time
        print(f"✓ Emotion state: {state_time:.3f}s for 100 updates ({100/state_time:.1f} ops/sec)")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling improvements"""
    print("\n=== Testing Error Handling ===")
    
    try:
        from emotune.core.emotion.fusion import EmotionFusion
        from emotune.core.emotion.emotion_state import EmotionState
        
        # Test fusion with invalid data
        fusion = EmotionFusion()
        
        # Test with None values
        result = fusion.fuse_emotions(None, None)
        print(f"✓ Fusion with None values: {result}")
        
        # Test with invalid emotion values
        invalid_face_data = {
            'emotions': {'valence': 2.0, 'arousal': -1.5},  # Out of range
            'confidence': 0.8
        }
        result = fusion.fuse_emotions(face_data=invalid_face_data, voice_data=None)
        print(f"✓ Fusion with invalid values: {result}")
        
        # Test emotion state with invalid data
        emotion_state = EmotionState()
        
        # Test with invalid covariance
        invalid_emotion_data = {
            'mean': {'valence': 0.5, 'arousal': 0.3},
            'covariance': "invalid",  # Not a valid covariance
            'uncertainty_trace': 0.3,
            'timestamp': time.time()
        }
        
        emotion_state.update_emotion(invalid_emotion_data)
        print("✓ Emotion state handled invalid covariance gracefully")
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_kalman_filter_fix():
    """Test the Kalman filter array ambiguity fix"""
    print("\n=== Testing Kalman Filter Fix ===")
    
    try:
        from emotune.core.emotion.kalman_filter import KalmanEmotionFilter
        
        kalman = KalmanEmotionFilter()
        
        # Test with valid observations
        observations = [
            {'valence': 0.5, 'arousal': 0.3, 'confidence': 0.8, 'uncertainty': 0.2},
            {'valence': 0.4, 'arousal': 0.4, 'confidence': 0.7, 'uncertainty': 0.3},
            {'valence': 0.6, 'arousal': 0.2, 'confidence': 0.9, 'uncertainty': 0.1}
        ]
        
        for i, obs in enumerate(observations):
            state, cov = kalman.update(obs)
            print(f"✓ Observation {i+1}: state={state[:2]}, cov_trace={np.trace(cov[:2,:2]):.3f}")
        
        # Test with invalid observations (should not crash)
        invalid_obs = {'valence': np.nan, 'arousal': 0.5, 'confidence': 0.5}
        state, cov = kalman.update(invalid_obs)
        print(f"✓ Invalid observation handled gracefully")
        
        return True
        
    except Exception as e:
        print(f"✗ Kalman filter fix test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_music_mapping_fix():
    """Test the music mapping attribute fix"""
    print("\n=== Testing Music Mapping Fix ===")
    
    try:
        from emotune.core.session.manager import SessionManager, SessionConfig
        from emotune.core.music.parameter_space import MusicParameterSpace
        
        # Create config
        config = SessionConfig()
        
        # Create parameter space
        param_space = MusicParameterSpace()
        
        # Create SessionManager (should not crash)
        session_manager = SessionManager(config, param_space=param_space)
        
        # Test music mapping
        if hasattr(session_manager, 'music_mapping'):
            params = session_manager.music_mapping.map_emotion_to_parameters(0.5, 0.3)
            print(f"✓ Music mapping works: {len(params)} parameters generated")
            return True
        else:
            print("✗ Music mapping attribute not found")
            return False
        
    except Exception as e:
        print(f"✗ Music mapping fix test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trajectory_progress_fix():
    """Test the trajectory progress tracking fix"""
    print("\n=== Testing Trajectory Progress Fix ===")
    
    try:
        from emotune.core.session.manager import SessionManager, SessionConfig
        from emotune.core.trajectory.library import TrajectoryType
        
        # Create config
        config = SessionConfig()
        
        # Create SessionManager
        session_manager = SessionManager(config)
        
        # Mock database
        mock_db = Mock()
        mock_db.get_emotion_history.return_value = [
            {'valence': 0.1, 'arousal': 0.6, 'timestamp': time.time() - 10},
            {'valence': 0.2, 'arousal': 0.5, 'timestamp': time.time() - 5},
            {'valence': 0.3, 'arousal': 0.4, 'timestamp': time.time()}
        ]
        session_manager.db = mock_db
        session_manager._current_session_id = "test_session"
        
        # Start trajectory
        session_manager.trajectory_planner.start_trajectory(TrajectoryType.CALM_DOWN, 300)
        
        # Test trajectory progress
        progress = session_manager._get_trajectory_progress()
        
        if progress and 'actual_path' in progress and 'target_path' in progress:
            print(f"✓ Trajectory progress works: {len(progress['actual_path'])} actual points, {len(progress['target_path'])} target points")
            return True
        else:
            print("✗ Trajectory progress not working")
            return False
        
    except Exception as e:
        print(f"✗ Trajectory progress fix test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_emotion_data_structure():
    """Test emotion data structure consistency"""
    print("\n=== Testing Emotion Data Structure ===")
    
    try:
        from emotune.core.emotion.fusion import EmotionFusion
        
        fusion = EmotionFusion()
        
        # Test with different emotion data formats
        face_data = {'valence': 0.5, 'arousal': 0.3, 'confidence': 0.8}
        voice_data = {'valence': 0.4, 'arousal': 0.4, 'confidence': 0.7}
        
        # Fuse emotions
        fused = fusion.fuse_emotions(face_data, voice_data)
        
        # Check structure
        if isinstance(fused, dict) and 'valence' in fused and 'arousal' in fused:
            print(f"✓ Emotion fusion works: valence={fused['valence']:.2f}, arousal={fused['arousal']:.2f}")
            return True
        else:
            print("✗ Emotion fusion structure incorrect")
            return False
        
    except Exception as e:
        print(f"✗ Emotion data structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_music_engine_fallback():
    """Test music engine fallback when Pyo unavailable"""
    print("\n=== Testing Music Engine Fallback ===")
    
    try:
        from emotune.core.music.engine_pyo import create_music_engine
        
        # Test engine creation
        engine = create_music_engine("auto")
        
        if engine:
            print(f"✓ Music engine created: {type(engine).__name__}")
            
            # Test parameter update
            params = {'tempo_bpm': 120, 'volume': 0.5}
            engine.update_parameters(params)
            print("✓ Parameter update works")
            
            return True
        else:
            print("✗ Music engine creation failed")
            return False
        
    except Exception as e:
        print(f"✗ Music engine fallback test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_fixes_tests():
    """Run all critical fix tests"""
    print("🧪 RUNNING CRITICAL FIXES TEST SUITE")
    print("=" * 50)
    
    tests = [
        test_kalman_filter_fix,
        test_music_mapping_fix,
        test_trajectory_progress_fix,
        test_emotion_data_structure,
        test_music_engine_fallback
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ {test.__name__} FAILED")
        except Exception as e:
            print(f"❌ {test.__name__} CRASHED: {e}")
    
    print("=" * 50)
    print(f"📊 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ ALL CRITICAL FIXES WORKING")
        return True
    else:
        print("❌ SOME CRITICAL FIXES STILL BROKEN")
        return False

def main():
    """Run all tests"""
    print("EmoTune Core Fixes Test Suite")
    print("=" * 50)
    
    tests = [
        ("Emotion Fusion", test_emotion_fusion),
        ("Emotion State", test_emotion_state),
        ("Kalman Filter", test_kalman_filter),
        ("Data Serialization", test_data_serialization),
        ("Performance", test_performance),
        ("Error Handling", test_error_handling)
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
        print("🎉 All tests passed! Core fixes are working correctly.")
    else:
        print("⚠ Some tests failed. Please review the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 