#!/usr/bin/env python3
"""
EmoTune Main Application
Run this file to start the complete EmoTune system
"""
import signal
import sys
import os
import json
from pathlib import Path
from types import SimpleNamespace
# from core.session.manager import SessionManager
from emotune.web.app import create_app
from emotune.utils.logging import setup_logging, get_logger
from emotune.utils.data_persistence import EmoTuneDatabase

# Load config
cfg_path = os.path.join(os.path.dirname(__file__), "config", "default.json")
with open(cfg_path) as f:
    cfg_dict = json.load(f)
config = SimpleNamespace(**cfg_dict)

# Handle Ctrl+C gracefully
def handle_interrupt(sig, frame):
    print("\n[EmoTune] KeyboardInterrupt received. Shutting down...")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_interrupt)

def main():
    # Setup logging - this returns the EmoTuneLogger instance
    emotune_logger = setup_logging()
    emotune_logger.set_session_id("main_session")  

    # Now you can use the logger directly
    emotune_logger.info("Starting EmoTune application...")
    emotune_logger.info("Initializing Database")
    
    try:
        db = EmoTuneDatabase()
    except Exception as e:
        emotune_logger.exception("Error initializing database")
        sys.exit(1)
    else:
        emotune_logger.info("Database initialized successfully")

    # Create Flask app
    app, socketio, session_manager = create_app(db)

    emotune_logger.info("Starting EmoTune server on http://localhost:5000")

    # Start Flask app with SocketIO
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        emotune_logger.exception("Error during server run")
    finally:
        emotune_logger.info("Shutting down EmoTune...")
        session_manager.shutdown()
        db.close()

if __name__ == "__main__":
    main()
