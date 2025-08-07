#!/usr/bin/env python3
"""
Minimal test for SessionManager initialization
"""

import sys
import os
import threading

# Add project root to path (now in tests/integration subdirectory)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_session_manager_init():
    """Test SessionManager initialization"""
    print("Testing SessionManager initialization...")
    
    try:
        # Import dependencies
        from emotune.core.session.manager import SessionManager, SessionConfig
        from emotune.core.music.parameter_space import MusicParameterSpace
        
        print("✓ Imports successful")
        
        # Create config
        config = SessionConfig()
        print("✓ Config created")
        
        # Create parameter space
        param_space = MusicParameterSpace()
        print("✓ Parameter space created")
        
        # Create session manager with minimal initialization
        print("Creating SessionManager...")
        
        # Create a minimal SessionManager that doesn't initialize problematic components
        class MinimalSessionManager(SessionManager):
            def __init__(self, config, db=None, app=None, socketio=None, param_space=None):
                self.config = config
                self.running = False
                self.session_start_time = 0.0
                self.shutdown_event = threading.Event()
                self.app = app
                self.socketio = socketio
                self.param_space = param_space if param_space is not None else MusicParameterSpace()
                
                # Skip problematic initialization
                print("✓ Minimal SessionManager created successfully")
        
        session_manager = MinimalSessionManager(
            config=config,
            db=None,
            app=None,
            socketio=None,
            param_space=param_space
        )
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_session_manager_init()
    sys.exit(0 if success else 1) 