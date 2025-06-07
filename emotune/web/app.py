import os
import json
import sys
from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
from types import SimpleNamespace
from datetime import datetime

from emotune.core.session.manager import SessionManager, SessionConfig
from emotune.utils.logging import get_logger
from emotune.config.trajectories import TRAJECTORY_TEMPLATES
from emotune.config.music_params import MUSIC_PARAMETERS

from dataclasses import fields

logger = get_logger()

# Load base config
logger.info("[app.py] Loading config...")
cfg_path = os.path.join(os.getcwd(), "config", "default.json")
with open(cfg_path) as f:
    cfg_dict = json.load(f)

session_cfg = SessionConfig(**cfg_dict.get('session', {}))
config = SimpleNamespace(**cfg_dict)
logger.info("[app.py] Config loaded.")

def create_app(db):
    logger.info("[create_app] STEP 1: Creating Flask app")
    app = Flask(__name__)
    app.config['SECRET_KEY'] = getattr(config, 'secret_key', 'emotune_secret_key')

    logger.info("[create_app] STEP 2: Creating SocketIO instance")
    socketio = SocketIO(app, async_mode='threading')

    logger.info("[create_app] STEP 3: Assigning database reference")
    data_persistence = db

    try:
        logger.info("[create_app] STEP 4: Initializing SessionManager")
        session_config_fields = {f.name for f in fields(SessionConfig)}
        filtered_session_cfg = {
            k: v for k, v in cfg_dict.get('session', {}).items() 
            if k in session_config_fields
        }
        session_cfg = SessionConfig(**filtered_session_cfg)
        
        session_manager = SessionManager(config=session_cfg, db=db, app=app, socketio=socketio)
        logger.info("[create_app] STEP 5: SessionManager initialized")
        logger.info("EmoTune system initialized successfully")
    except Exception as e:
        logger.error(f"[create_app] ERROR during SessionManager init: {e}")
        session_manager = None

    logger.info("[create_app] STEP 6: Registering routes")

    @app.route('/')
    def index():
        logger.info("[route /] Dashboard requested")
        return render_template('dashboard.html')

    @app.route('/favicon.ico')
    def favicon():
        return app.send_static_file('favicon.ico')


    @app.route('/session/start', methods=['POST'])
    def start_session():
        data = request.get_json()
        trajectory_type = data.get('trajectory_type', 'calm_down')
        duration = int(data.get('duration', 300))

        try:
            session_id = session_manager.start_session(trajectory_type, duration)
            session['session_id'] = session_id
            # Emit session status to frontend
            socketio.emit('session_status', {'active': True, 'session_id': session_id, 'trajectory_type': trajectory_type, 'duration': duration})
            return jsonify({
                'success': True,
                'session_id': session_id,
                'message': 'Session started'
            })
        except Exception as e:
            socketio.emit('error', {'message': str(e)})
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/session/stop', methods=['POST'])
    def stop_session():
        sid = session.get('session_id')
        if not sid:
            return jsonify({'success': False, 'error': 'No active session'}), 400

        try:
            session_manager.stop_session(sid)
            session.pop('session_id', None)
            # Emit session status to frontend
            socketio.emit('session_status', {'active': False, 'session_id': sid})
            return jsonify({'success': True, 'message': 'Session stopped'})
        except Exception as e:
            socketio.emit('error', {'message': str(e)})
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/emotion/history/<session_id>')
    def get_emotion_history(session_id):
        logger.info(f"[GET /emotion/history/{session_id}] Called")
        try:
            history = data_persistence.get_emotion_history(session_id)
            return jsonify({'success': True, 'history': history})
        except Exception as e:
            logger.error(f"Failed to get emotion history: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/trajectories')
    def get_trajectories():
        logger.info("[GET /trajectories] Called")
        try:
            trajs = [
                {'name': n, 'type': t['type'], 'duration': t['duration'], 'description': t['description']}
                for n, t in TRAJECTORY_TEMPLATES.items()
            ]
            return jsonify({'success': True, 'trajectories': trajs})
        except Exception as e:
            logger.error(f"Failed to get trajectories: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/feedback', methods=['POST'])
    def submit_feedback():
        logger.info("[POST /feedback] Called")
        try:
            data = request.get_json()
            sid = session.get('session_id')
            if not sid:
                return jsonify({'success': False, 'error': 'No active session'}), 400
            feedback = {
                'session_id': sid,
                'timestamp': datetime.now().isoformat(),
                'rating': data.get('rating'),
                'comfort': data.get('comfort'),
                'effectiveness': data.get('effectiveness'),
                'comments': data.get('comments', ''),
                'emotion_state': data.get('emotion_state')
            }
            session_manager.process_feedback(sid, feedback)
            data_persistence.save_feedback(sid, feedback)
            logger.info(f"[POST /feedback] Feedback saved for session: {sid}")
            return jsonify({'success': True, 'message': 'Feedback submitted'})
        except Exception as e:
            logger.error(f"Failed to submit feedback: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    # --- SESSION UUID TO SOCKETIO SID MAPPING ---
    session_uuid_to_sid = {}
    sid_to_session_uuid = {}

    @socketio.on('connect')
    def on_connect():
        logger.info(f"[SocketIO] Client connected: {request.sid}")
        emit('connected', {'message': 'Connected'})

    @socketio.on('disconnect')
    def on_disconnect():
        logger.info(f"[SocketIO] Client disconnected: {request.sid}")
        # Remove from mapping if present
        uuid_to_remove = None
        for uuid, sid in session_uuid_to_sid.items():
            if sid == request.sid:
                uuid_to_remove = uuid
                break
        if uuid_to_remove:
            del session_uuid_to_sid[uuid_to_remove]
            del sid_to_session_uuid[request.sid]

    @socketio.on('start_emotion_monitoring')
    def on_start_monitoring():
        logger.info("[SocketIO] Emotion monitoring requested")
        sid = session.get('session_id')
        if sid:
            # Map session UUID to SocketIO sid
            session_uuid_to_sid[sid] = request.sid
            sid_to_session_uuid[request.sid] = sid
            # Set the real SocketIO sid in SessionManager for backend emits
            session_manager.set_socketio_sid(request.sid)
            session_manager.start_emotion_monitoring(sid)
            emit('monitoring_started', {'message': 'Monitoring started'})
        else:
            emit('error', {'message': 'No active session'})

    @socketio.on('emotion_data')
    def on_emotion_data(data):
        logger.info("[SocketIO] Emotion data received: %s", data)
        try:
            # --- PATCH: Extract session_uuid from payload and set sid mapping ---
            session_uuid = data.get('session_uuid') if isinstance(data, dict) else None
            if session_uuid:
                session_uuid_to_sid[session_uuid] = request.sid
                sid_to_session_uuid[request.sid] = session_uuid
                session_manager.set_socketio_sid(request.sid)
                sid = session_uuid
            else:
                sid = session.get('session_id')
            logger.info(f"[SocketIO] Processing emotion data for session: {sid}")
            res = session_manager.process_emotion_data(sid, data)
            logger.info(f"[SocketIO] process_emotion_data result: {res}")
            # Get trajectory progress for visualization
            info = session_manager.trajectory_planner.get_trajectory_info()
            logger.info(f"[SocketIO] Trajectory info: {info}")
            actual_path = data_persistence.get_emotion_history(sid)
            logger.info(f"[SocketIO] Actual emotion path: {actual_path}")
            actual_points = [
                {'valence': p['mean']['valence'], 'arousal': p['mean']['arousal'], 'timestamp': p['timestamp']}
                for p in actual_path
            ] if actual_path else []
            if actual_points and info['target']:
                logger.info(f"[SocketIO] Actual points and target present. Generating target points.")
                target_fn = session_manager.trajectory_planner.current_trajectory
                start_time = session_manager.trajectory_planner.start_time
                target_points = []
                for p in actual_points:
                    try:
                        result = target_fn(p['timestamp'] - start_time)
                        logger.info(f"[SocketIO] target_fn({p['timestamp'] - start_time}) returned: {result} (type: {type(result)})")
                        if isinstance(result, (list, tuple)) and len(result) >= 2:
                            valence, arousal = result[0], result[1]
                        else:
                            logger.error(f"[SocketIO] target_fn did not return a tuple/list: {result}")
                            valence, arousal = None, None
                        target_points.append({'valence': valence, 'arousal': arousal, 'timestamp': p['timestamp']})
                    except Exception as e:
                        logger.error(f"[SocketIO] Error calling target_fn: {e}")
                        target_points.append({'valence': None, 'arousal': None, 'timestamp': p['timestamp']})
            else:
                logger.info(f"[SocketIO] Actual points or target missing. No target points generated.")
                target_points = []
            deviation = info.get('deviation', None)
            logger.info(f"[SocketIO] Deviation: {deviation}")
            logger.info(f"[SocketIO] Emitting emotion_update event with music_parameters: {res.get('music_parameters')}")
            # Use the real SocketIO sid for emits
            socketio_sid = session_uuid_to_sid.get(sid, request.sid)
            # Emit a single event with all relevant data
            socketio.emit('emotion_update', {
                'emotion_state': res['emotion_state'],
                'trajectory_progress': {
                    'info': info,
                    'actual_path': actual_points,
                    'target_path': target_points,
                    'deviation': deviation
                },
                'music_parameters': res['music_parameters'],
                'timestamp': datetime.now().isoformat()
            }, room=socketio_sid)
        except Exception as e:
            logger.error(f"Error processing emotion data: {e}")
            emit('error', {'message': 'Processing failed'})

    @socketio.on('feedback_data')
    def on_feedback_data(data):
        logger.info("[SocketIO] Feedback data received")
        try:
            sid = session.get('session_id')
            session_manager.process_realtime_feedback(sid, data)
            emit('feedback_processed', {'message': 'Feedback processed'})
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            emit('error', {'message': 'Processing failed'})

    @app.route('/api/status')
    def api_status():
        logger.info("[GET /api/status] Called")
        try:
            status = {
                'system_online': session_manager is not None,
                'active_sessions': session_manager.get_active_session_count(),
                'server_time': datetime.now().isoformat(),
                'version': getattr(config, 'version', '1.0.0')
            }
            return jsonify({'success': True, 'status': status})
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/settings', methods=['GET', 'POST'])
    def api_settings():
        logger.info(f"[API /settings] Method: {request.method}")
        if request.method == 'GET':
            try:
                settings = data_persistence.get_settings()
                return jsonify({'success': True, 'settings': settings})
            except Exception as e:
                logger.error(f"Failed to get settings: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        else:
            try:
                new = request.get_json()
                data_persistence.save_settings(new)
                session_manager.update_settings(new)
                return jsonify({'success': True, 'message': 'Settings updated'})
            except Exception as e:
                logger.error(f"Failed to update settings: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/trajectory_progress')
    def api_trajectory_progress():
        sid = session.get('session_id')
        if not sid:
            return jsonify({'success': False, 'error': 'No active session'}), 400
        try:
            # Get current trajectory info and actual/target paths
            info = session_manager.trajectory_planner.get_trajectory_info()
            # Optionally, add actual and target paths for visualization
            actual_path = data_persistence.get_emotion_history(sid)
            # Format for frontend: list of {valence, arousal, timestamp}
            actual_points = [
                {'valence': p['mean']['valence'], 'arousal': p['mean']['arousal'], 'timestamp': p['timestamp']}
                for p in actual_path
            ] if actual_path else []
            # Generate target path for the same timestamps
            if actual_points and info['target']:
                target_fn = session_manager.trajectory_planner.current_trajectory
                start_time = session_manager.trajectory_planner.start_time
                target_points = [
                    {'valence': target_fn(p['timestamp'] - start_time)[0],
                     'arousal': target_fn(p['timestamp'] - start_time)[1],
                     'timestamp': p['timestamp']}
                    for p in actual_points
                ]
            else:
                target_points = []
            return jsonify({'success': True, 'progress': {
                'info': info,
                'actual_path': actual_points,
                'target_path': target_points
            }})
        except Exception as e:
            logger.error(f"Failed to get trajectory progress: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    @socketio.on('music_control')
    def on_music_control(data):
        logger.info(f"[SocketIO] Music control: {data}")
        sid = session.get('session_id')
        if not sid:
            emit('error', {'message': 'No active session'})
            return
        action = data.get('action')
        if action == 'play':
            session_manager.play_music()
            emit('music_status', {'status': 'playing'})
        elif action == 'pause':
            session_manager.pause_music()
            emit('music_status', {'status': 'paused'})
        elif action == 'regenerate':
            session_manager.regenerate_music()
            emit('music_status', {'status': 'regenerated'})
        else:
            emit('error', {'message': 'Unknown music control action'})

    @socketio.on('request_music_parameters')
    def on_request_music_parameters():
        sid = session.get('session_id')
        if not sid:
            emit('error', {'message': 'No active session'})
            return
        params = session_manager.music_renderer.current_params
        emit('music_parameters', params)

    @socketio.on('request_trajectory_info')
    def on_request_trajectory_info():
        sid = session.get('session_id')
        if not sid:
            emit('error', {'message': 'No active session'})
            return
        info = session_manager.trajectory_planner.get_trajectory_info()
        emit('trajectory_info', info)

    @socketio.on('session_ready')
    def handle_session_ready(data):
        sid = session.get('session_id')
        logger.info(f"[SocketIO] session_ready received for session: {sid}")
        # Optionally, set a ready flag or perform any setup
        emit('session_ready_ack', {'session_id': sid})

    @app.errorhandler(404)
    def not_found(e):
        logger.warning("[ERROR 404] Not found")
        return render_template('error.html', error='Page not found'), 404

    @app.errorhandler(500)
    def internal_error(e):
        logger.error(f"[ERROR 500] Internal error: {e}")
        return render_template('error.html', error='Internal error'), 500

    logger.info("[create_app] STEP 7: App setup complete")
    return app, socketio, session_manager
