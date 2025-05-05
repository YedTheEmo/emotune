from flask import Flask
from .emotion_service import EmotionService
from .music_engine.music_controller import MusicController


def create_app():
    app = Flask(__name__)
    emotion_service = EmotionService()
    music_controller = MusicController()

    # Initialize routes with both emotion_service and music_controller
    from .routes import init_routes
    init_routes(app, emotion_service, music_controller)

    # Start background threads
    emotion_service.start_background_analysis()
    music_controller.start_music_loop(emotion_service)

    return app

