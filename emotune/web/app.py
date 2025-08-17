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
    
    # Initialize session tracking variables
    session_uuid_to_sid = {}
    sid_to_session_uuid = {}

    # Try to initialize SessionManager, but fall back gracefully
    session_manager = None
    try:
        logger.info("[create_app] STEP 4: Initializing SessionManager")
        session_config_fields = {f.name for f in fields(SessionConfig)}
        filtered_session_cfg = {
            k: v for k, v in cfg_dict.get('session', {}).items() 
            if k in session_config_fields
        }
        session_cfg = SessionConfig(**filtered_session_cfg)
        
        # Create parameter space for SessionManager
        from emotune.core.music.parameter_space import MusicParameterSpace
        param_space = MusicParameterSpace()
        
        # Try to import and create SessionManager
        try:
            from emotune.core.session.manager import SessionManager
            # Use positional arguments to avoid any parameter issues
            session_manager = SessionManager(session_cfg, db, app, socketio, param_space)
        except Exception as main_error:
            logger.error(f"Main SessionManager failed: {main_error}")
            # Try working SessionManager
            from emotune.core.session.working_manager import WorkingSessionManager
            session_manager = WorkingSessionManager(session_cfg, db, app, socketio, param_space)
        logger.info("[create_app] STEP 5: SessionManager initialized successfully")
        logger.info("EmoTune system initialized successfully")
    except Exception as e:
        logger.error(f"[create_app] ERROR during SessionManager init: {e}")
        logger.error(f"[create_app] Creating fallback SessionManager")
        # Create a fallback SessionManager
        try:
            from emotune.core.session.fallback_manager import FallbackSessionManager
            session_manager = FallbackSessionManager(config=session_cfg, db=db, app=app, socketio=socketio, param_space=param_space)
            logger.info("[create_app] Fallback SessionManager initialized successfully")
        except Exception as fallback_error:
            logger.error(f"[create_app] Fallback SessionManager also failed: {fallback_error}")
            session_manager = None

    logger.info("[create_app] STEP 6: Registering routes")

    @app.route('/')
    def index():
        logger.info("[route /] Dashboard requested")
        return render_template('dashboard.html')

    @app.route('/favicon.ico')
    def favicon():
        return app.send_static_file('favicon.ico')

    # --- MJPEG video feed from backend capture ---
    def _generate_mjpeg_frames():
        import cv2
        import time
        while True:
            try:
                if session_manager and hasattr(session_manager, 'emotion_capture') and session_manager.emotion_capture:
                    frame = session_manager.emotion_capture.get_last_frame()
                else:
                    frame = None
                if frame is None:
                    time.sleep(0.05)
                    continue
                ok, buf = cv2.imencode('.jpg', frame)
                if not ok:
                    time.sleep(0.01)
                    continue
                jpg = buf.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
            except GeneratorExit:
                break
            except Exception:
                time.sleep(0.05)

    @app.route('/video_feed')
    def video_feed():
        from flask import Response
        return Response(_generate_mjpeg_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


    @app.route('/session/start', methods=['POST'])
    def start_session():
        try:
            if not session_manager:
                return jsonify({
                    'success': False,
                    'error': 'Session manager not available'
                }), 500
                
            data = request.get_json()
            trajectory_type = data.get('trajectory_type', 'calm_down')
            duration = int(data.get('duration', 300))

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
            logger.error(f"Error starting session: {e}")
            socketio.emit('error', {'message': str(e)})
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/session/stop', methods=['POST'])
    def stop_session():
        sid = session.get('session_id')
        logger.info(f"[stop_session] Flask session contains: {dict(session)}")
        logger.info(f"[stop_session] Looking for session_id, found: {sid}")
        
        if not sid:
            # Check if we have a session manager with a current session
            if session_manager and hasattr(session_manager, '_current_session_id') and session_manager._current_session_id:
                sid = session_manager._current_session_id
                logger.info(f"[stop_session] Using session_id from session_manager: {sid}")
            elif session_manager and hasattr(session_manager, 'current_session_id') and session_manager.current_session_id:
                sid = session_manager.current_session_id
                logger.info(f"[stop_session] Using current_session_id from session_manager: {sid}")
            else:
                logger.warning("[stop_session] No active session found in Flask session or session manager")
                return jsonify({'success': False, 'error': 'No active session'}), 400

        try:
            session_manager.stop_session(sid)
            session.pop('session_id', None)
            # Emit session status to frontend
            socketio.emit('session_status', {'active': False, 'session_id': sid})
            return jsonify({'success': True, 'message': 'Session stopped'})
        except Exception as e:
            logger.error(f"Error stopping session: {e}")
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
            session_manager.process_realtime_feedback(sid, feedback)
            logger.info(f"[POST /feedback] Feedback processed for session: {sid}")
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
        # Set the SocketIO SID immediately when client connects
        session_manager.set_socketio_sid(request.sid)
        emit('connected', {'message': 'Connected to EmoTune server'})

    @socketio.on('disconnect')
    def on_disconnect():
        logger.info(f"[SocketIO] Client disconnected: {request.sid}")
        # Clear the SocketIO SID when client disconnects
        session_manager.set_socketio_sid(None)

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
            
            # Extract actual emotion data from the payload
            emotion_data = data.get('emotion_data', data) if isinstance(data, dict) else data
            
            # Ensure we have proper emotion data structure
            if isinstance(emotion_data, (int, float)):
                # If we got a timestamp, create default emotion structure
                logger.warning(f"Received timestamp instead of emotion data: {emotion_data}")
                emotion_data = {
                    'mean': {'valence': 0.0, 'arousal': 0.5},
                    'confidence': 0.5,
                    'timestamp': emotion_data
                }
            elif not isinstance(emotion_data, dict):
                # If we got something unexpected, create default structure
                logger.warning(f"Received unexpected emotion data type: {type(emotion_data)}")
                emotion_data = {
                    'mean': {'valence': 0.0, 'arousal': 0.5},
                    'confidence': 0.5,
                    'timestamp': time.time()
                }
            
            res = session_manager.process_emotion_data(sid, emotion_data)
            logger.info(f"[SocketIO] process_emotion_data result: {res}")
            
            # Get trajectory progress for visualization
            try:
                info = session_manager.trajectory_planner.get_trajectory_info()
                logger.info(f"[SocketIO] Trajectory info: {info}")
            except Exception as e:
                logger.error(f"Error getting trajectory info: {e}")
                info = {'target': None, 'deviation': 0.0}
            
            try:
                actual_path = data_persistence.get_emotion_history(sid)
            except Exception as e:
                logger.error(f"Error getting emotion history: {e}")
                actual_path = []
            logger.info(f"[SocketIO] Actual emotion path: {actual_path}")
            actual_points = [
                {'valence': p['mean']['valence'], 'arousal': p['mean']['arousal'], 'timestamp': p['timestamp']}
                for p in actual_path
            ] if actual_path else []
            if actual_points and info.get('target'):
                logger.info(f"[SocketIO] Actual points and target present. Generating target points.")
                try:
                    target_fn = session_manager.trajectory_planner.current_trajectory
                    start_time = session_manager.trajectory_planner.start_time
                except Exception as e:
                    logger.error(f"Error getting trajectory function: {e}")
                    target_fn = lambda t: (0.0, 0.0)
                    start_time = time.time()
                target_points = []
                for p in actual_points:
                    try:
                        result = target_fn(p['timestamp'] - start_time)
                        logger.debug(f"[SocketIO] target_fn({p['timestamp'] - start_time}) returned: {result}")
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
            # Get trajectory progress from session manager
            try:
                trajectory_progress = session_manager._get_trajectory_progress()
                deviation = trajectory_progress.get('deviation', None)
                logger.info(f"[SocketIO] Deviation from session manager: {deviation}")
            except Exception as e:
                logger.error(f"Error getting trajectory progress: {e}")
                deviation = None
            
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
        try:
            if not session_manager:
                emit('error', {'message': 'Session manager not available'})
                return
            
            # Allow music controls even without active session
            sid = session.get('session_id')
            if not sid:
                logger.info("[SocketIO] No active session, but allowing music control")
                
            action = data.get('action')
            if action == 'play':
                try:
                    session_manager.play_music()
                    emit('music_status', {'status': 'playing'})
                except Exception as e:
                    logger.error(f"Error playing music: {e}")
                    emit('error', {'message': f'Failed to play music: {str(e)}'})
            elif action == 'pause':
                try:
                    session_manager.pause_music()
                    emit('music_status', {'status': 'paused'})
                except Exception as e:
                    logger.error(f"Error pausing music: {e}")
                    emit('error', {'message': f'Failed to pause music: {str(e)}'})
            elif action == 'regenerate':
                try:
                    session_manager.regenerate_music()
                    emit('music_status', {'status': 'regenerated'})
                except Exception as e:
                    logger.error(f"Error regenerating music: {e}")
                    emit('error', {'message': f'Failed to regenerate music: {str(e)}'})
            else:
                emit('error', {'message': 'Unknown music control action'})
        except Exception as e:
            logger.error(f"Error in music control: {e}")
            emit('error', {'message': f'Music control error: {str(e)}'})

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

    @socketio.on('update_confidence_thresholds')
    def on_update_confidence_thresholds(data):
        logger.info(f"[SocketIO] Confidence thresholds received: {data}")
        if session_manager:
            session_manager.update_confidence_thresholds(data)
            emit('thresholds_updated', {'message': 'Confidence thresholds updated successfully', 'success': True})
        else:
            emit('error', {'message': 'Session manager not available'})

    @socketio.on('update_analysis_mode')
    def on_update_analysis_mode(data):
        logger.info(f"[SocketIO] Analysis mode received: {data}")
        if session_manager:
            session_manager.update_analysis_mode(data.get('mode'))
            emit('mode_updated', {'message': 'Analysis mode updated successfully', 'success': True})
        else:
            emit('error', {'message': 'Session manager not available'})

    @socketio.on('update_fusion_options')
    def on_update_fusion_options(data):
        logger.info(f"[SocketIO] Fusion options received: {data}")
        if session_manager:
            session_manager.update_fusion_options(data)
            emit('fusion_options_updated', {'message': 'Fusion options updated successfully', 'success': True})
        else:
            emit('error', {'message': 'Session manager not available'})

    @socketio.on('start_session')
    def on_start_session(data):
        trajectory = data.get('trajectory', 'default')
        duration = data.get('duration', 300)
        sid = request.sid
        session_id = session_manager.start_session(trajectory_type=trajectory, duration=duration)
        session['session_id'] = session_id
        logger.info(f"[SocketIO] Session started: {session_id} for SID: {sid}")
        session_manager.play_music() # Start music automatically
        emit('session_started', {'session_id': session_id})

    @socketio.on('stop_session')
    def on_stop_session():
        sid = session.get('session_id')
        logger.info(f"[SocketIO] Stopping session: {sid}")
        if sid:
            session_manager.stop_session(sid)
            emit('session_stopped', {'session_id': sid})
            session.pop('session_id', None)
        else:
            logger.warning("[SocketIO] No active session to stop.")

    @socketio.on('play_music')
    def on_play_music():
        logger.info("[SocketIO] Play music request received")
        if session_manager:
            session_manager.play_music()

    @socketio.on('pause_music')
    def on_pause_music():
        logger.info("[SocketIO] Pause music request received")
        if session_manager:
            session_manager.pause_music()

    @socketio.on('regenerate_music')
    def on_regenerate_music():
        logger.info("[SocketIO] Regenerate music request received")
        if session_manager:
            session_manager.regenerate_music()

    @socketio.on('submit_feedback')
    def on_submit_feedback(data):
        session_id = session.get('session_id')
        logger.info(f"[SocketIO] Submitting feedback for session: {session_id}")
        if not session_id:
            emit('error', {'message': 'No active session'})
            return
        try:
            feedback = {
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'rating': data.get('rating'),
                'comfort': data.get('comfort'),
                'effectiveness': data.get('effectiveness'),
                'comments': data.get('comments', ''),
                'emotion_state': data.get('emotion_state')
            }
            session_manager.process_realtime_feedback(session_id, feedback)
            logger.info(f"[SocketIO] Feedback processed for session: {session_id}")
            emit('feedback_submitted', {'message': 'Feedback submitted'})
        except Exception as e:
            logger.error(f"[SocketIO] Failed to submit feedback: {e}")
            emit('error', {'message': f'Failed to submit feedback: {str(e)}'})

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
