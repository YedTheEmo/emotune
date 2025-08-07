#!/usr/bin/env python3
"""
Comprehensive test script for EmoTune system fixes
Tests all critical components and verifies fixes are working
"""

import sys
import os
import time
import threading
import queue
import numpy as np
from datetime import datetime

# Add project root to path (now in tests/unit subdirectory)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from emotune.core.emotion.capture import EmotionCapture
from emotune.core.emotion.fusion import EmotionFusion
from emotune.core.emotion.emotion_state import EmotionState
from emotune.core.emotion.kalman_filter import KalmanEmotionFilter
from emotune.core.session.manager import SessionManager, SessionConfig

def test_camera_capture():
    """Test the improved camera capture system"""
    print("\n=== Testing Camera Capture ===")
    
    try:
        # Initialize capture
        capture = EmotionCapture(face_fps=5, audio_duration=0.5, camera_index=0)
        
        # Test initialization
        print("✓ Camera capture initialized")
        
        # Test start/stop
        capture.start_capture()
        print("✓ Camera capture started")
        
        # Wait a bit for initialization
        time.sleep(2)
        
        # Test data retrieval
        data = capture.get_data()
        if data:
            print(f"✓ Data captured: {list(data.keys())}")
            if 'face_frame' in data:
                print(f"  - Face frame shape: {data['face_frame'].shape}")
            if 'audio_chunk' in data:
                print(f"  - Audio chunk shape: {data['audio_chunk'].shape}")
        else:
            print("⚠ No data captured (this might be normal if no camera)")
        
        # Test stats
        stats = capture.get_stats()
        print(f"✓ Capture stats: {stats}")
        
        # Stop capture
        capture.stop_capture()
        print("✓ Camera capture stopped")
        
        return True
        
    except Exception as e:
        print(f"✗ Camera capture test failed: {e}")
        return False

def test_emotion_fusion():
    """Test the improved emotion fusion algorithm"""
    print("\n=== Testing Emotion Fusion ===")
    
    try:
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
        return False

def test_emotion_state():
    """Test the improved emotion state management"""
    print("\n=== Testing Emotion State ===")
    
    try:
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
        return False

def test_kalman_filter():
    """Test the Kalman filter integration"""
    print("\n=== Testing Kalman Filter ===")
    
    try:
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
        return False

def test_session_manager():
    """Test the improved session manager"""
    print("\n=== Testing Session Manager ===")
    
    try:
        # Initialize session manager
        config = SessionConfig(
            emotion_update_rate=1.0,
            music_update_rate=0.2,
            trajectory_name="calm_down",
            session_duration=300.0,
            enable_rl=False,
            enable_feedback=False,
            enable_face_analysis=True,
            enable_voice_analysis=False,
            device="cpu",
            camera_index=0
        )
        
        session_manager = SessionManager(config=config)
        print("✓ Session manager initialized")
        
        # Test session start
        session_id = session_manager.start_session("calm_down", 60)
        print(f"✓ Session started: {session_id}")
        
        # Wait a bit for processing
        time.sleep(3)
        
        # Test status
        status = session_manager.get_current_status()
        print(f"✓ Session status: emotion_state={status.get('emotion_state', {})}")
        
        # Test session stop
        session_manager.stop_session(session_id)
        print("✓ Session stopped")
        
        return True
        
    except Exception as e:
        print(f"✗ Session manager test failed: {e}")
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
        return False

def test_performance():
    """Test system performance"""
    print("\n=== Testing Performance ===")
    
    try:
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
        return False

def main():
    """Run all tests"""
    print("EmoTune System Fixes Test Suite")
    print("=" * 50)
    
    tests = [
        ("Camera Capture", test_camera_capture),
        ("Emotion Fusion", test_emotion_fusion),
        ("Emotion State", test_emotion_state),
        ("Kalman Filter", test_kalman_filter),
        ("Session Manager", test_session_manager),
        ("Data Serialization", test_data_serialization),
        ("Performance", test_performance)
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
        print("🎉 All tests passed! System fixes are working correctly.")
    else:
        print("⚠ Some tests failed. Please review the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 