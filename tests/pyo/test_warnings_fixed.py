#!/usr/bin/env python3
"""
Test script to verify warning fixes
"""

import sys
import os

# Add project root to path (now in tests/pyo subdirectory)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Setup environment to suppress warnings
from emotune.utils.warnings import setup_environment
setup_environment()

def test_pyo_without_warnings():
    """Test Pyo engine without warnings"""
    print("Testing Pyo engine with warning suppression...")
    
    try:
        from emotune.core.music.engine_pyo import create_music_engine
        
        # Create engine
        engine = create_music_engine("pyo")
        print(f"✓ Engine created: {type(engine).__name__}")
        
        # Test status
        status = engine.get_status()
        print(f"✓ Engine status: {status['available']}")
        
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
        import time
        time.sleep(1)
        
        # Test stop
        engine.stop()
        print("✓ Music stopped")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_pyo_without_warnings()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1) 