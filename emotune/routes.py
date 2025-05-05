from flask import render_template, jsonify, request
import traceback
import numpy as np

def init_routes(app, emotion_service, music_controller):
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/emotion_data')
    def emotion_data():
        try:
            data = emotion_service.get_raw_data()

            def convert_to_native(data):
                if isinstance(data, np.generic):
                    return float(data)
                elif isinstance(data, dict):
                    return {key: convert_to_native(value) for key, value in data.items()}
                return data

            data = convert_to_native(data)

            return jsonify({
                'valence': data.get('valence', 0.0),
                'arousal': data.get('arousal', 0.0),
                'face': data.get('face', {}),
                'voice': data.get('voice', {})
            })

        except Exception as e:
            print("Error in emotion_data route:", e)
            return jsonify({'error': str(e)}), 500

    @app.route('/play', methods=['POST'])
    def play_music():
        try:
            data = request.get_json()
            valence = float(data.get('valence', 0))
            arousal = float(data.get('arousal', 0))
            music_controller.play_music(valence, arousal)
            return jsonify({'status': 'playing', 'valence': valence, 'arousal': arousal})
        except Exception as e:
            return jsonify({'error': str(e)}), 400

    @app.route('/stop')
    def stop_music():
        music_controller.stop_music()
        return jsonify({'status': 'stopped'})
