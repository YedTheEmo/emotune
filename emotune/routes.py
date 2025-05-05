from flask import render_template, jsonify
import traceback
from .emotion_service import EmotionService
import numpy as np

# Function to initialize routes
def init_routes(app, emotion_service: EmotionService):
    # Route for the root URL to serve index.html
    @app.route('/')
    def index():
        return render_template('index.html')  # Serve the main HTML page

    # Route for the emotion data endpoint
    @app.route('/emotion_data')
    def emotion_data():
        try:
            # Fetch emotion data from the emotion service
            data = emotion_service.get_raw_data()

            # Log the data to confirm it's being returned correctly
            print("Raw Data from Emotion Service:", data)

            # Ensure values are converted to native Python types (float, int, etc.)
            def convert_to_native(data):
                if isinstance(data, np.generic):  # Catch numpy types like float32
                    return float(data)  # Convert numpy float to standard Python float
                elif isinstance(data, dict):  # If it's a dictionary, recursively convert it
                    return {key: convert_to_native(value) for key, value in data.items()}
                return data  # If it's already a simple type, return it as is

            # Convert all values in the data to standard Python types
            data = convert_to_native(data)

            # Return emotion data as JSON
            return jsonify({
                'emotion': data.get('emotion', 'Neutral'),
                'valence': data.get('valence', 0.0),
                'arousal': data.get('arousal', 0.0),
                'face': {
                    'valence': data.get('face', {}).get('valence', 0.0),
                    'arousal': data.get('face', {}).get('arousal', 0.0)
                },
                'voice': {
                    'valence': data.get('voice', {}).get('valence', 0.0),
                    'arousal': data.get('voice', {}).get('arousal', 0.0)
                }
            })
        except Exception as e:
            # Print the full traceback to the console for detailed error info
            print("Error in emotion_data route:", e)
            traceback.print_exc()

            # Return error details in JSON format with traceback
            return jsonify({
                'error': str(e),
                'traceback': traceback.format_exc()
            }), 500

