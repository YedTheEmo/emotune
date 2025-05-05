#from flask import Flask
#from emotune.routes import init_routes
#from emotune.emotion_service import EmotionService
#
#emotion_service = EmotionService()  # Single shared instance
#emotion_service.start_background_analysis()
#
#def create_app():
    #app = Flask(__name__)
    #init_routes(app, emotion_service)  # Pass it in!
    #return app
#


from flask import Flask
from .emotion_service import EmotionService

def create_app():
    app = Flask(__name__)
    emotion_service = EmotionService()
    
    # Initialize routes with the emotion service
    from .routes import init_routes
    init_routes(app, emotion_service)
    
    # Start the background analysis thread
    emotion_service.start_background_analysis()
    
    return app
