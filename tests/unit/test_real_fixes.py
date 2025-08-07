import numpy as np
import time
from unittest.mock import Mock, patch

def test_real_emotion_processing():
    """Test the complete emotion processing pipeline with real data"""
    print("\n=== Testing Real Emotion Processing Pipeline ===")
    
    try:
        from emotune.core.session.manager import SessionManager, SessionConfig
        from emotune.core.music.parameter_space import MusicParameterSpace
        
        # Create config
        config = SessionConfig()
        
        # Create parameter space
        param_space = MusicParameterSpace()
        
        # Create SessionManager
        session_manager = SessionManager(config, param_space=param_space)
        
        # Test with real emotion data (like from logs)
        test_emotion_data = {
            'session_uuid': 'test-session-123',
            'emotion_data': {
                'timestamp': time.time(),
                'confidence': 0.5,
                'mean': {'valence': 0.0, 'arousal': 0.5}
            }
        }
        
        # Process emotion data
        result = session_manager.process_emotion_data('test-session-123', test_emotion_data['emotion_data'])
        
        # Check that emotion state is properly serialized
        emotion_state = result.get('emotion_state', {})
        assert 'valence' in emotion_state, "Missing valence in emotion state"
        assert 'arousal' in emotion_state, "Missing arousal in emotion state"
        assert 'confidence' in emotion_state, "Missing confidence in emotion state"
        
        # Check that values are not (0,0) - should be actual values
        valence = emotion_state['valence']
        arousal = emotion_state['arousal']
        print(f"✓ Emotion state: valence={valence:.3f}, arousal={arousal:.3f}")
        
        assert valence == 0.0, f"Expected valence=0.0, got {valence}"
        assert arousal == 0.5, f"Expected arousal=0.5, got {arousal}"
        
        # Check that music parameters are generated
        music_params = result.get('music_parameters', {})
        assert len(music_params) > 0, "No music parameters generated"
        
        # Check that parameters are not static defaults
        tempo = music_params.get('tempo_bpm', 0)
        brightness = music_params.get('brightness', 0)
        dissonance = music_params.get('dissonance_level', 0)
        
        print(f"✓ Music parameters: tempo={tempo:.0f}, brightness={brightness:.2f}, dissonance={dissonance:.2f}")
        
        # Parameters should be different from static defaults for this emotion
        assert tempo != 100.0, f"Tempo should not be static default 100, got {tempo}"
        assert brightness != 0.5, f"Brightness should not be static default 0.5, got {brightness}"
        assert dissonance != 0.2, f"Dissonance should not be static default 0.2, got {dissonance}"
        
        # Check trajectory progress
        trajectory_progress = result.get('trajectory_progress', {})
        assert 'deviation' in trajectory_progress, "Missing deviation in trajectory progress"
        
        deviation = trajectory_progress['deviation']
        print(f"✓ Trajectory deviation: {deviation:.3f}")
        
        assert isinstance(deviation, (int, float)), f"Deviation should be numeric, got {type(deviation)}"
        assert 0 <= deviation <= 2, f"Deviation should be in reasonable range, got {deviation}"
        
        return True
        
    except Exception as e:
        print(f"✗ Real emotion processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_music_parameter_serialization():
    """Test that music parameters are properly serialized without numpy types"""
    print("\n=== Testing Music Parameter Serialization ===")
    
    try:
        from emotune.core.session.manager import SessionManager, SessionConfig
        from emotune.core.music.parameter_space import MusicParameterSpace
        
        # Create config
        config = SessionConfig()
        
        # Create parameter space
        param_space = MusicParameterSpace()
        
        # Create SessionManager
        session_manager = SessionManager(config, param_space=param_space)
        
        # Test with numpy types (like what would come from parameter generation)
        test_params = {
            'tempo_bpm': np.float64(120.0),
            'brightness': np.float32(0.7),
            'dissonance_level': np.array([0.3])[0],
            'voice_density': np.int32(3),
            'overall_volume': 0.8
        }
        
        # Serialize parameters
        serialized = session_manager._serialize_music_parameters(test_params)
        
        # Check that all values are Python types
        for key, value in serialized.items():
            assert isinstance(value, (int, float, str, list)), f"Parameter {key} has type {type(value)}, should be Python type"
        
        print(f"✓ Serialized {len(serialized)} parameters successfully")
        
        # Check specific values
        assert serialized['tempo_bpm'] == 120.0, f"Tempo not preserved: {serialized['tempo_bpm']}"
        assert serialized['brightness'] == 0.7, f"Brightness not preserved: {serialized['brightness']}"
        assert serialized['dissonance_level'] == 0.3, f"Dissonance not preserved: {serialized['dissonance_level']}"
        assert serialized['voice_density'] == 3, f"Voice density not preserved: {serialized['voice_density']}"
        
        return True
        
    except Exception as e:
        print(f"✗ Music parameter serialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_emotion_state_serialization_real():
    """Test emotion state serialization with real data structures"""
    print("\n=== Testing Emotion State Serialization Real ===")
    
    try:
        from emotune.core.session.manager import SessionManager, SessionConfig
        
        # Create config
        config = SessionConfig()
        
        # Create SessionManager
        session_manager = SessionManager(config)
        
        # Test with real emotion data structure (like from logs)
        test_emotion_data = {
            'mean': {'valence': 0.0, 'arousal': 0.5},
            'confidence': 0.5,
            'timestamp': time.time()
        }
        
        serialized = session_manager._serialize_emotion_state(test_emotion_data)
        
        # Check structure
        assert 'valence' in serialized, "Missing valence in serialized data"
        assert 'arousal' in serialized, "Missing arousal in serialized data"
        assert 'confidence' in serialized, "Missing confidence in serialized data"
        assert 'mean' in serialized, "Missing mean structure in serialized data"
        assert 'timestamp' in serialized, "Missing timestamp in serialized data"
        
        # Check values
        valence = serialized['valence']
        arousal = serialized['arousal']
        confidence = serialized['confidence']
        
        print(f"✓ Serialized: valence={valence:.2f}, arousal={arousal:.2f}, confidence={confidence:.2f}")
        
        # Verify values match input
        assert valence == 0.0, f"Valence should be 0.0, got {valence}"
        assert arousal == 0.5, f"Arousal should be 0.5, got {arousal}"
        assert confidence == 0.5, f"Confidence should be 0.5, got {confidence}"
        
        # Check mean structure
        assert serialized['mean']['valence'] == 0.0, "Mean valence should be 0.0"
        assert serialized['mean']['arousal'] == 0.5, "Mean arousal should be 0.5"
        
        return True
        
    except Exception as e:
        print(f"✗ Emotion state serialization real test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trajectory_deviation_real():
    """Test trajectory deviation calculation with real data"""
    print("\n=== Testing Trajectory Deviation Real ===")
    
    try:
        from emotune.core.session.manager import SessionManager, SessionConfig
        
        # Create config
        config = SessionConfig()
        
        # Create SessionManager
        session_manager = SessionManager(config)
        
        # Mock database with real data (like from logs)
        mock_db = Mock()
        mock_db.get_emotion_history.return_value = [
            {'valence': 0.0, 'arousal': 0.5, 'timestamp': time.time() - 10},
            {'valence': 0.0, 'arousal': 0.5, 'timestamp': time.time() - 5},
            {'valence': 0.0, 'arousal': 0.5, 'timestamp': time.time()}
        ]
        session_manager.db = mock_db
        session_manager._current_session_id = "test_session"
        
        # Start trajectory
        from emotune.core.trajectory.library import TrajectoryType
        session_manager.trajectory_planner.start_trajectory(TrajectoryType.CALM_DOWN, 300)
        
        # Test trajectory progress
        progress = session_manager._get_trajectory_progress()
        
        # Check structure
        assert 'actual_path' in progress, "Missing actual_path in progress"
        assert 'target_path' in progress, "Missing target_path in progress"
        assert 'deviation' in progress, "Missing deviation in progress"
        assert 'current_target' in progress, "Missing current_target in progress"
        
        # Check values
        deviation = progress['deviation']
        actual_count = len(progress['actual_path'])
        target_count = len(progress['target_path'])
        
        print(f"✓ Deviation: {deviation:.3f}, actual points: {actual_count}, target points: {target_count}")
        
        # Verify deviation is reasonable
        assert 0 <= deviation <= 2, f"Deviation {deviation} out of reasonable range"
        assert actual_count > 0, "No actual points in trajectory"
        assert target_count > 0, "No target points in trajectory"
        
        # Check that deviation is not None
        assert deviation is not None, "Deviation should not be None"
        
        return True
        
    except Exception as e:
        print(f"✗ Trajectory deviation real test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_real_fixes_tests():
    """Run all real fixes tests"""
    print("🧪 RUNNING REAL FIXES VERIFICATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_real_emotion_processing,
        test_music_parameter_serialization,
        test_emotion_state_serialization_real,
        test_trajectory_deviation_real
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
    
    print("=" * 60)
    print(f"📊 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ ALL REAL FIXES VERIFIED WORKING")
        return True
    else:
        print("❌ SOME REAL FIXES STILL NEED ATTENTION")
        return False

if __name__ == "__main__":
    run_all_real_fixes_tests() 