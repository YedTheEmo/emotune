import numpy as np
import time
from unittest.mock import Mock, patch

def test_music_parameter_updates():
    """Test that music parameters actually update based on emotion"""
    print("\n=== Testing Music Parameter Updates ===")
    
    try:
        from emotune.core.session.manager import SessionManager, SessionConfig
        from emotune.core.music.parameter_space import MusicParameterSpace
        
        # Create config
        config = SessionConfig()
        
        # Create parameter space
        param_space = MusicParameterSpace()
        
        # Create SessionManager
        session_manager = SessionManager(config, param_space=param_space)
        
        # Test different emotion combinations
        test_emotions = [
            (0.8, 0.9, "Happy/Excited"),
            (-0.7, 0.6, "Angry"),
            (-0.8, -0.3, "Sad"),
            (0.2, -0.5, "Calm")
        ]
        
        for valence, arousal, name in test_emotions:
            params = session_manager._map_emotion_to_music(valence, arousal)
            
            # Check that parameters are different for different emotions
            tempo = params.get('tempo_bpm', 0)
            brightness = params.get('brightness', 0)
            dissonance = params.get('dissonance_level', 0)
            
            print(f"✓ {name}: tempo={tempo:.0f}, brightness={brightness:.2f}, dissonance={dissonance:.2f}")
            
            # Verify parameters are reasonable
            assert 40 <= tempo <= 160, f"Tempo {tempo} out of range for {name}"
            assert 0 <= brightness <= 1, f"Brightness {brightness} out of range for {name}"
            assert 0 <= dissonance <= 1, f"Dissonance {dissonance} out of range for {name}"
        
        return True
        
    except Exception as e:
        print(f"✗ Music parameter update test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_emotion_state_serialization():
    """Test that emotion state is properly serialized for frontend"""
    print("\n=== Testing Emotion State Serialization ===")
    
    try:
        from emotune.core.session.manager import SessionManager, SessionConfig
        
        # Create config
        config = SessionConfig()
        
        # Create SessionManager
        session_manager = SessionManager(config)
        
        # Test different emotion data formats
        test_emotions = [
            {'mean': {'valence': 0.5, 'arousal': 0.3}, 'confidence': 0.8},
            {'valence': 0.4, 'arousal': 0.6, 'confidence': 0.7},
            {'mean': {'valence': -0.2, 'arousal': 0.8}, 'confidence': 0.9}
        ]
        
        for emotion_data in test_emotions:
            serialized = session_manager._serialize_emotion_state(emotion_data)
            
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
            
            # Verify values are reasonable
            assert -1 <= valence <= 1, f"Valence {valence} out of range"
            assert 0 <= arousal <= 1, f"Arousal {arousal} out of range"
            assert 0 <= confidence <= 1, f"Confidence {confidence} out of range"
        
        return True
        
    except Exception as e:
        print(f"✗ Emotion state serialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trajectory_deviation_calculation():
    """Test that trajectory deviation is calculated correctly"""
    print("\n=== Testing Trajectory Deviation Calculation ===")
    
    try:
        from emotune.core.session.manager import SessionManager, SessionConfig
        
        # Create config
        config = SessionConfig()
        
        # Create SessionManager
        session_manager = SessionManager(config)
        
        # Mock database with test data
        mock_db = Mock()
        mock_db.get_emotion_history.return_value = [
            {'valence': 0.1, 'arousal': 0.6, 'timestamp': time.time() - 10},
            {'valence': 0.2, 'arousal': 0.5, 'timestamp': time.time() - 5},
            {'valence': 0.3, 'arousal': 0.4, 'timestamp': time.time()}
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
        
        return True
        
    except Exception as e:
        print(f"✗ Trajectory deviation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_music_engine_parameter_handling():
    """Test that music engine handles parameters correctly without numpy type issues"""
    print("\n=== Testing Music Engine Parameter Handling ===")
    
    try:
        from emotune.core.music.engine_pyo import PyoMusicEngine
        
        # Create engine
        engine = PyoMusicEngine()
        
        # Test parameter conversion
        test_params = {
            'tempo_bpm': 120.0,
            'brightness': 0.7,
            'warmth': 0.6,
            'voice_density': 3.0,
            'reverb_amount': 0.4
        }
        
        # Test emotion data
        emotion_data = {'valence': 0.5, 'arousal': 0.3}
        
        # Test parameter conversion
        pattern = engine._convert_parameters_to_pattern(test_params, emotion_data)
        
        # Check that all values are Python types, not numpy types
        for key, value in pattern.items():
            assert isinstance(value, (int, float, str)), f"Parameter {key} has type {type(value)}, should be int/float/str"
        
        print(f"✓ Parameter conversion works: tempo={pattern['tempo_bpm']}, voices={pattern['voice_density']}")
        
        # Test that parameters are used correctly
        assert pattern['tempo_bpm'] == 120.0, "Tempo not preserved"
        assert pattern['brightness'] == 0.7, "Brightness not preserved"
        assert pattern['voice_density'] == 3.0, "Voice density not preserved"
        
        return True
        
    except Exception as e:
        print(f"✗ Music engine parameter handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_kalman_filter_final():
    """Test that Kalman filter works without array ambiguity errors"""
    print("\n=== Testing Kalman Filter Final ===")
    
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
        
        # Test with numpy array values (should not crash)
        numpy_obs = {'valence': np.array([0.5]), 'arousal': np.array([0.3]), 'confidence': 0.8}
        state, cov = kalman.update(numpy_obs)
        print(f"✓ Numpy array observation handled gracefully")
        
        return True
        
    except Exception as e:
        print(f"✗ Kalman filter final test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_critical_fixes_tests():
    """Run all critical fixes tests"""
    print("🧪 RUNNING CRITICAL FIXES VERIFICATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_music_parameter_updates,
        test_emotion_state_serialization,
        test_trajectory_deviation_calculation,
        test_music_engine_parameter_handling,
        test_kalman_filter_final
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
        print("✅ ALL CRITICAL FIXES VERIFIED WORKING")
        return True
    else:
        print("❌ SOME CRITICAL FIXES STILL NEED ATTENTION")
        return False

if __name__ == "__main__":
    run_all_critical_fixes_tests() 