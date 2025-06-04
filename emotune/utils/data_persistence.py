"""
Data Persistence Module for EmoTune
Handles session data storage, user preferences, and model state persistence
"""

import pickle
import json
import sqlite3
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
import numpy as np
import uuid

@dataclass
class UserProfile:
    """User profile data structure"""
    user_id: str
    created_at: datetime
    preferences: Dict[str, Any]
    emotion_baselines: Dict[str, float]  # Personal emotion baselines
    music_preferences: Dict[str, float]  # Preferred parameter ranges
    trajectory_history: List[str]  # Previously used trajectories
    total_sessions: int = 0
    total_duration_minutes: float = 0.0
    last_active: Optional[datetime] = None

@dataclass
class SessionData:
    """Complete session data structure"""
    session_id: str
    user_id: Optional[str]
    start_time: datetime
    end_time: Optional[datetime]
    trajectory_type: Optional[str]
    initial_emotion: Optional[Tuple[float, float]]  # (valence, arousal)
    final_emotion: Optional[Tuple[float, float]]
    trajectory_completion: Optional[float]
    user_ratings: List[float]
    music_parameters_history: List[Dict[str, float]]
    emotion_history: List[Tuple[float, float, float]]  # (timestamp, valence, arousal)
    feedback_events: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class EmoTuneDatabase:
    """SQLite database handler for EmoTune data persistence"""
    
    def __init__(self, db_path: Union[str, Path] = "emotune.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._local = threading.local()
        self._init_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0
            )
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection
    
    def _init_database(self):
        """Initialize database tables"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                preferences TEXT,
                emotion_baselines TEXT,
                music_preferences TEXT,
                trajectory_history TEXT,
                total_sessions INTEGER DEFAULT 0,
                total_duration_minutes REAL DEFAULT 0.0,
                last_active TEXT
            )
        ''')
        
        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                start_time TEXT NOT NULL,
                end_time TEXT,
                trajectory_type TEXT,
                initial_valence REAL,
                initial_arousal REAL,
                final_valence REAL,
                final_arousal REAL,
                trajectory_completion REAL,
                metadata TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Emotion data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS emotion_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                valence REAL NOT NULL,
                arousal REAL NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                raw_data TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        ''')
        
        # Music parameters table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS music_parameters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                parameters TEXT NOT NULL,
                trajectory_progress REAL,
                dtw_error REAL,
                generation_time_ms REAL,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        ''')
        
        # Feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                feedback_type TEXT NOT NULL,
                rating REAL,
                category TEXT,
                context TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        ''')
        
        # Model states table (for RL agent persistence)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                model_type TEXT NOT NULL,
                state_data BLOB NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Create indexes for better query performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions (user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_emotion_data_session ON emotion_data (session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_music_parameters_session ON music_parameters (session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback (session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_states_user ON model_states (user_id, model_type)')
        
        conn.commit()
    
    def create_user(self, preferences: Optional[Dict] = None) -> str:
        """Create a new user profile"""
        user_id = str(uuid.uuid4())
        now = datetime.now()
        
        profile = UserProfile(
            user_id=user_id,
            created_at=now,
            preferences=preferences or {},
            emotion_baselines={},
            music_preferences={},
            trajectory_history=[]
        )
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO users (
                user_id, created_at, preferences, emotion_baselines,
                music_preferences, trajectory_history, total_sessions,
                total_duration_minutes, last_active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            profile.user_id,
            profile.created_at.isoformat(),
            json.dumps(profile.preferences),
            json.dumps(profile.emotion_baselines),
            json.dumps(profile.music_preferences),
            json.dumps(profile.trajectory_history),
            profile.total_sessions,
            profile.total_duration_minutes,
            profile.last_active.isoformat() if profile.last_active else None
        ))
        
        conn.commit()
        return user_id
    
    def get_user(self, user_id: str) -> Optional[UserProfile]:
        """Retrieve user profile"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        return UserProfile(
            user_id=row['user_id'],
            created_at=datetime.fromisoformat(row['created_at']),
            preferences=json.loads(row['preferences']),
            emotion_baselines=json.loads(row['emotion_baselines']),
            music_preferences=json.loads(row['music_preferences']),
            trajectory_history=json.loads(row['trajectory_history']),
            total_sessions=row['total_sessions'],
            total_duration_minutes=row['total_duration_minutes'],
            last_active=datetime.fromisoformat(row['last_active']) if row['last_active'] else None
        )
    
    def update_user(self, profile: UserProfile):
        """Update user profile"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users SET
                preferences = ?, emotion_baselines = ?, music_preferences = ?,
                trajectory_history = ?, total_sessions = ?, total_duration_minutes = ?,
                last_active = ?
            WHERE user_id = ?
        ''', (
            json.dumps(profile.preferences),
            json.dumps(profile.emotion_baselines),
            json.dumps(profile.music_preferences),
            json.dumps(profile.trajectory_history),
            profile.total_sessions,
            profile.total_duration_minutes,
            profile.last_active.isoformat() if profile.last_active else None,
            profile.user_id
        ))
        
        conn.commit()
    
    def create_session(self, user_id: Optional[str] = None, 
                      trajectory_type: Optional[str] = None) -> str:
        """Create a new session"""
        session_id = str(uuid.uuid4())
        now = datetime.now()
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sessions (
                session_id, user_id, start_time, trajectory_type, metadata
            ) VALUES (?, ?, ?, ?, ?)
        ''', (
            session_id,
            user_id,
            now.isoformat(),
            trajectory_type,
            json.dumps({})
        ))
        
        conn.commit()
        return session_id
    
    def end_session(self, session_id: str, 
                   final_emotion: Optional[Tuple[float, float]] = None,
                   trajectory_completion: Optional[float] = None):
        """End a session and update statistics"""
        now = datetime.now()
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Update session
        cursor.execute('''
            UPDATE sessions SET
                end_time = ?, final_valence = ?, final_arousal = ?,
                trajectory_completion = ?
            WHERE session_id = ?
        ''', (
            now.isoformat(),
            final_emotion[0] if final_emotion else None,
            final_emotion[1] if final_emotion else None,
            trajectory_completion,
            session_id
        ))
        
        # Get session info for user statistics update
        cursor.execute('''
            SELECT user_id, start_time FROM sessions WHERE session_id = ?
        ''', (session_id,))
        
        session_row = cursor.fetchone()
        if session_row and session_row['user_id']:
            user_id = session_row['user_id']
            start_time = datetime.fromisoformat(session_row['start_time'])
            duration_minutes = (now - start_time).total_seconds() / 60
            
            # Update user statistics
            cursor.execute('''
                UPDATE users SET
                    total_sessions = total_sessions + 1,
                    total_duration_minutes = total_duration_minutes + ?,
                    last_active = ?
                WHERE user_id = ?
            ''', (duration_minutes, now.isoformat(), user_id))
        
        conn.commit()
    
    def save_emotion_data(self, session_id: str, timestamp: float,
                         valence: float, arousal: float, confidence: float,
                         source: str, raw_data: Optional[Dict] = None):
        """Save emotion detection data"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO emotion_data (
                session_id, timestamp, valence, arousal, confidence, source, raw_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id, timestamp, valence, arousal, confidence, source,
            json.dumps(raw_data) if raw_data else None
        ))
        
        conn.commit()
    
    def save_music_parameters(self, session_id: str, timestamp: float,
                            parameters: Dict[str, float],
                            trajectory_progress: Optional[float] = None,
                            dtw_error: Optional[float] = None,
                            generation_time_ms: Optional[float] = None):
        """Save music generation parameters"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO music_parameters (
                session_id, timestamp, parameters, trajectory_progress,
                dtw_error, generation_time_ms
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            session_id, timestamp, json.dumps(parameters),
            trajectory_progress, dtw_error, generation_time_ms
        ))
        
        conn.commit()
    
    def save_feedback(self, session_id: str, timestamp: float,
                     feedback_type: str, rating: Optional[float] = None,
                     category: Optional[str] = None,
                     context: Optional[Dict] = None):
        """Save user feedback"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO feedback (
                session_id, timestamp, feedback_type, rating, category, context
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            session_id, timestamp, feedback_type, rating, category,
            json.dumps(context) if context else None
        ))
        
        conn.commit()
    
    def get_session_data(self, session_id: str) -> Optional[SessionData]:
        """Retrieve complete session data"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get session info
        cursor.execute('SELECT * FROM sessions WHERE session_id = ?', (session_id,))
        session_row = cursor.fetchone()
        
        if not session_row:
            return None
        
        # Get emotion data
        cursor.execute('''
            SELECT timestamp, valence, arousal FROM emotion_data
            WHERE session_id = ? ORDER BY timestamp
        ''', (session_id,))
        emotion_history = [(row['timestamp'], row['valence'], row['arousal']) 
                          for row in cursor.fetchall()]
        
        # Get music parameters
        cursor.execute('''
            SELECT parameters FROM music_parameters
            WHERE session_id = ? ORDER BY timestamp
        ''', (session_id,))
        music_parameters_history = [json.loads(row['parameters']) 
                                   for row in cursor.fetchall()]
        
        # Get feedback
        cursor.execute('''
            SELECT timestamp, feedback_type, rating, category, context
            FROM feedback WHERE session_id = ? ORDER BY timestamp
        ''', (session_id,))
        feedback_events = []
        for row in cursor.fetchall():
            feedback = {
                'timestamp': row['timestamp'],
                'feedback_type': row['feedback_type'],
                'rating': row['rating'],
                'category': row['category']
            }
            if row['context']:
                feedback['context'] = json.loads(row['context'])
            feedback_events.append(feedback)
        
        # Get user ratings
        user_ratings = [fb['rating'] for fb in feedback_events 
                       if fb['rating'] is not None]
        
        return SessionData(
            session_id=session_row['session_id'],
            user_id=session_row['user_id'],
            start_time=datetime.fromisoformat(session_row['start_time']),
            end_time=datetime.fromisoformat(session_row['end_time']) if session_row['end_time'] else None,
            trajectory_type=session_row['trajectory_type'],
            initial_emotion=(session_row['initial_valence'], session_row['initial_arousal']) 
                           if session_row['initial_valence'] is not None else None,
            final_emotion=(session_row['final_valence'], session_row['final_arousal'])
                         if session_row['final_valence'] is not None else None,
            trajectory_completion=session_row['trajectory_completion'],
            user_ratings=user_ratings,
            music_parameters_history=music_parameters_history,
            emotion_history=emotion_history,
            feedback_events=feedback_events,
            metadata=json.loads(session_row['metadata']) if session_row['metadata'] else {}
        )
    
    def get_user_sessions(self, user_id: str, limit: int = 50) -> List[SessionData]:
        """Get recent sessions for a user"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT session_id FROM sessions
            WHERE user_id = ?
            ORDER BY start_time DESC
            LIMIT ?
        ''', (user_id, limit))
        
        session_ids = [row['session_id'] for row in cursor.fetchall()]
        return [self.get_session_data(sid) for sid in session_ids if self.get_session_data(sid)]
    
    def save_model_state(self, user_id: Optional[str], model_type: str, 
                        state_data: Any, metadata: Optional[Dict] = None):
        """Save model state (e.g., RL agent weights)"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        serialized_data = pickle.dumps(state_data)
        now = datetime.now()
        
        # Delete old state for this user/model combination
        cursor.execute('''
            DELETE FROM model_states WHERE user_id = ? AND model_type = ?
        ''', (user_id, model_type))
        
        # Insert new state
        cursor.execute('''
            INSERT INTO model_states (
                user_id, model_type, state_data, created_at, metadata
            ) VALUES (?, ?, ?, ?, ?)
        ''', (
            user_id, model_type, serialized_data, now.isoformat(),
            json.dumps(metadata) if metadata else None
        ))
        
        conn.commit()
    
    def load_model_state(self, user_id: Optional[str], model_type: str) -> Optional[Any]:
        """Load model state"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT state_data FROM model_states
            WHERE user_id = ? AND model_type = ?
            ORDER BY created_at DESC
            LIMIT 1
        ''', (user_id, model_type))
        
        row = cursor.fetchone()
        if row:
            return pickle.loads(row['state_data'])
        return None
    
    def get_user_emotion_history(self, user_id: str, days: int = 30) -> List[Tuple[datetime, float, float]]:
        """Get user's emotion history over specified days"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute('''
            SELECT ed.timestamp, ed.valence, ed.arousal
            FROM emotion_data ed
            JOIN sessions s ON ed.session_id = s.session_id
            WHERE s.user_id = ? AND s.start_time >= ?
            ORDER BY ed.timestamp
        ''', (user_id, cutoff_date))
        
        return [(datetime.fromtimestamp(row['timestamp']), row['valence'], row['arousal'])
                for row in cursor.fetchall()]
    
    def get_trajectory_analytics(self, user_id: str, trajectory_type: str) -> Dict[str, Any]:
        """Get analytics for a specific trajectory type for a user"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_sessions,
                AVG(trajectory_completion) as avg_completion,
                AVG(CASE WHEN trajectory_completion >= 0.8 THEN 1.0 ELSE 0.0 END) as success_rate,
                AVG((julianday(end_time) - julianday(start_time)) * 24 * 60) as avg_duration_minutes
            FROM sessions
            WHERE user_id = ? AND trajectory_type = ? AND end_time IS NOT NULL
        ''', (user_id, trajectory_type))
        
        row = cursor.fetchone()
        
        # Get average user ratings for this trajectory
        cursor.execute('''
            SELECT AVG(f.rating) as avg_rating
            FROM feedback f
            JOIN sessions s ON f.session_id = s.session_id
            WHERE s.user_id = ? AND s.trajectory_type = ? AND f.rating IS NOT NULL
        ''', (user_id, trajectory_type))
        
        rating_row = cursor.fetchone()
        
        return {
            'total_sessions': row['total_sessions'] or 0,
            'avg_completion': row['avg_completion'] or 0.0,
            'success_rate': row['success_rate'] or 0.0,
            'avg_duration_minutes': row['avg_duration_minutes'] or 0.0,
            'avg_rating': rating_row['avg_rating'] or 0.0
        }
    
    def cleanup_old_sessions(self, days_to_keep: int = 90):
        """Clean up old session data to manage database size"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
        
        # Get sessions to delete
        cursor.execute('''
            SELECT session_id FROM sessions WHERE start_time < ?
        ''', (cutoff_date,))
        
        old_session_ids = [row['session_id'] for row in cursor.fetchall()]
        
        if not old_session_ids:
            return 0
        
        # Delete related data
        placeholders = ','.join(['?' for _ in old_session_ids])
        
        cursor.execute(f'DELETE FROM emotion_data WHERE session_id IN ({placeholders})', old_session_ids)
        cursor.execute(f'DELETE FROM music_parameters WHERE session_id IN ({placeholders})', old_session_ids)
        cursor.execute(f'DELETE FROM feedback WHERE session_id IN ({placeholders})', old_session_ids)
        cursor.execute(f'DELETE FROM sessions WHERE session_id IN ({placeholders})', old_session_ids)
        
        conn.commit()
        return len(old_session_ids)
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        stats = {}
        tables = ['users', 'sessions', 'emotion_data', 'music_parameters', 'feedback', 'model_states']
        
        for table in tables:
            cursor.execute(f'SELECT COUNT(*) as count FROM {table}')
            stats[table] = cursor.fetchone()['count']
        
        return stats
    
    def close(self):
        """Close database connections"""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
    
    def get_emotion_history(self, session_id: str) -> list:
        """Return emotion history for a session as a list of dicts with mean valence/arousal/timestamp."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT timestamp, valence, arousal, confidence, source, raw_data
            FROM emotion_data WHERE session_id = ? ORDER BY timestamp
        ''', (session_id,))
        history = []
        for row in cursor.fetchall():
            # For compatibility with app.py, wrap val/arousal in 'mean' dict
            history.append({
                'timestamp': row['timestamp'],
                'mean': {
                    'valence': row['valence'],
                    'arousal': row['arousal']
                },
                'confidence': row['confidence'],
                'source': row['source'],
                'raw_data': json.loads(row['raw_data']) if row['raw_data'] else None
            })
        return history


# Utility functions for data analysis
def calculate_emotion_trends(emotion_history: List[Tuple[datetime, float, float]]) -> Dict[str, float]:
    """Calculate emotion trends from history"""
    if len(emotion_history) < 2:
        return {'valence_trend': 0.0, 'arousal_trend': 0.0}
    
    # Simple linear trend calculation
    times = [(e[0] - emotion_history[0][0]).total_seconds() for e in emotion_history]
    valences = [e[1] for e in emotion_history]
    arousals = [e[2] for e in emotion_history]
    
    valence_trend = np.polyfit(times, valences, 1)[0] if len(times) > 1 else 0.0
    arousal_trend = np.polyfit(times, arousals, 1)[0] if len(times) > 1 else 0.0
    
    return {
        'valence_trend': float(valence_trend * 3600),  # Per hour
        'arousal_trend': float(arousal_trend * 3600)   # Per hour
    }


def export_user_data(db: EmoTuneDatabase, user_id: str, output_path: str):
    """Export all user data to JSON file"""
    user_profile = db.get_user(user_id)
    if not user_profile:
        raise ValueError(f"User {user_id} not found")
    
    sessions = db.get_user_sessions(user_id, limit=1000)
    emotion_history = db.get_user_emotion_history(user_id, days=365)
    
    export_data = {
        'user_profile': asdict(user_profile),
        'sessions': [asdict(session) for session in sessions],
        'emotion_history': [(e[0].isoformat(), e[1], e[2]) for e in emotion_history],
        'exported_at': datetime.now().isoformat()
    }
    
    # Convert datetime objects to strings for JSON serialization
    def datetime_converter(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2, default=datetime_converter)


# Example usage and testing
if __name__ == "__main__":
    # Initialize database
    db = EmoTuneDatabase("test_emotune.db")
    
    # Create a test user
    user_id = db.create_user({
        'preferred_genres': ['ambient', 'classical'],
        'max_session_duration': 30
    })
    print(f"Created user: {user_id}")
    
    # Create a test session
    session_id = db.create_session(user_id, "calm_down")
    print(f"Created session: {session_id}")
    
    # Save some test data
    import time
    current_time = time.time()
    
    db.save_emotion_data(session_id, current_time, 0.3, -0.2, 0.8, "face_analysis")
    db.save_music_parameters(session_id, current_time, {
        'tempo': 80,
        'harmony_complexity': 0.4,
        'texture_density': 0.3
    }, trajectory_progress=0.1)
    
    db.save_feedback(session_id, current_time, "rating", rating=4.0, category="overall")
    
    # End session
    db.end_session(session_id, final_emotion=(0.6, -0.4), trajectory_completion=0.85)
    
    # Retrieve session data
    session_data = db.get_session_data(session_id)
    print(f"Session completion: {session_data.trajectory_completion}")
    
    # Get analytics
    analytics = db.get_trajectory_analytics(user_id, "calm_down")
    print(f"Trajectory analytics: {analytics}")
    
    # Database stats
    stats = db.get_database_stats()
    print(f"Database stats: {stats}")
    
    db.close()
