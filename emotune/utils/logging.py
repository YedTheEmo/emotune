"""
Enhanced Logging System for EmoTune
Provides structured logging with performance metrics, session tracking, and analysis capabilities
"""

import logging
import json
import time
import threading
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import numpy as np

@dataclass
class EmotionLogEntry:
    """Structured log entry for emotion data"""
    timestamp: float
    session_id: str
    valence: float
    arousal: float
    confidence: float
    source: str  # 'face', 'voice', 'fused'
    raw_data: Optional[Dict] = None

@dataclass
class MusicLogEntry:
    """Structured log entry for music generation"""
    timestamp: float
    session_id: str
    parameters: Dict[str, float]
    trajectory_type: Optional[str] = None
    trajectory_progress: Optional[float] = None
    dtw_error: Optional[float] = None
    generation_time_ms: Optional[float] = None

@dataclass
class FeedbackLogEntry:
    """Structured log entry for user feedback"""
    timestamp: float
    session_id: str
    feedback_type: str  # 'explicit', 'implicit'
    rating: Optional[float] = None
    category: Optional[str] = None  # 'like', 'dislike', 'skip'
    context: Optional[Dict] = None

@dataclass
class PerformanceLogEntry:
    """Structured log entry for system performance"""
    timestamp: float
    session_id: str
    component: str
    operation: str
    duration_ms: float
    memory_mb: Optional[float] = None
    cpu_percent: Optional[float] = None
    gpu_utilization: Optional[float] = None

class EmoTuneLogger:
    """Enhanced logger for EmoTune with structured data and analytics"""


    def __init__(self,
                 log_dir: Union[str, Path] = "logs",
                 log_level: int = logging.INFO,
                 max_file_size_mb: int = 100,
                 backup_count: int = 5,
                 enable_performance_logging: bool = True):

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)  # ADDED parents=True

        self.enable_performance_logging = enable_performance_logging
        self._setup_loggers(log_level, max_file_size_mb, backup_count)
        self._setup_structured_loggers()

        # Thread-local storage for session context
        self._local = threading.local()
        self.set_session_id("main_thread")  # SET DEFAULT SESSION ID

        # Performance tracking
        self._operation_timers = {}
        self._performance_stats = {}

        self.logger.info("EmoTune logging system initialized")


    def _setup_loggers(self, log_level: int, max_file_size_mb: int, backup_count: int):
        """Setup standard Python loggers"""
        from logging.handlers import RotatingFileHandler

        # Main application logger
        self.logger = logging.getLogger('emotune')
        self.logger.setLevel(logging.DEBUG)  # Always allow all logs to handlers

        # Clear existing handlers
        self.logger.handlers = []

        # File handler with rotation
        file_handler = None
        try:
            file_handler = RotatingFileHandler(
                self.log_dir / 'emotune.log',
                maxBytes=max_file_size_mb * 1024 * 1024,
                backupCount=backup_count
            )
        except Exception as e:
            print(f"File handler error: {e}", file=sys.stderr)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        if file_handler:
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.DEBUG)  # File gets all logs
            self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)  # Console gets only INFO+
        self.logger.addHandler(console_handler)

        # Prevent log propagation to root logger (avoids duplicate/extra output)
        self.logger.propagate = False

    def _setup_structured_loggers(self):
        """Setup structured data loggers for analytics"""
        self.emotion_log_file = self.log_dir / 'emotions.jsonl'
        self.music_log_file = self.log_dir / 'music.jsonl'
        self.feedback_log_file = self.log_dir / 'feedback.jsonl'
        self.performance_log_file = self.log_dir / 'performance.jsonl'

        # Removed eager file creation to avoid empty files
        # Files will be created lazily on first write

    def set_session_id(self, session_id: str):
        """Set session ID for current thread"""
        self._local.session_id = session_id
        self.logger.info(f"Session started: {session_id}")

    def get_session_id(self) -> str:
        """Get current session ID"""
        return getattr(self._local, 'session_id', 'unknown')

    def log_emotion(self,
                   valence: float,
                   arousal: float,
                   confidence: float,
                   source: str,
                   raw_data: Optional[Dict] = None):
        """Log emotion detection data"""
        entry = EmotionLogEntry(
            timestamp=time.time(),
            session_id=self.get_session_id(),
            valence=valence,
            arousal=arousal,
            confidence=confidence,
            source=source,
            raw_data=raw_data
        )

        self._write_structured_log(self.emotion_log_file, entry)
        self.logger.info(f"Emotion logged: {source} V={valence:.3f} A={arousal:.3f} C={confidence:.3f}")

    def log_music_generation(self,
                           parameters: Dict[str, float],
                           trajectory_type: Optional[str] = None,
                           trajectory_progress: Optional[float] = None,
                           dtw_error: Optional[float] = None,
                           generation_time_ms: Optional[float] = None):
        """Log music generation parameters and metrics"""
        entry = MusicLogEntry(
            timestamp=time.time(),
            session_id=self.get_session_id(),
            parameters=parameters,
            trajectory_type=trajectory_type,
            trajectory_progress=trajectory_progress,
            dtw_error=dtw_error,
            generation_time_ms=generation_time_ms
        )

        self._write_structured_log(self.music_log_file, entry)
        self.logger.info(f"Music generation logged: {len(parameters)} params, "
                         f"trajectory={trajectory_type}, DTW={dtw_error}")

    def log_feedback(self,
                    feedback_type: str,
                    rating: Optional[float] = None,
                    category: Optional[str] = None,
                    context: Optional[Dict] = None):
        """Log user feedback"""
        entry = FeedbackLogEntry(
            timestamp=time.time(),
            session_id=self.get_session_id(),
            feedback_type=feedback_type,
            rating=rating,
            category=category,
            context=context
        )

        self._write_structured_log(self.feedback_log_file, entry)
        self.logger.info(f"Feedback logged: {feedback_type} - {category} - {rating}")

    def log_performance(self,
                       component: str,
                       operation: str,
                       duration_ms: float,
                       memory_mb: Optional[float] = None,
                       cpu_percent: Optional[float] = None,
                       gpu_utilization: Optional[float] = None):
        """Log performance metrics"""
        if not self.enable_performance_logging:
            return

        entry = PerformanceLogEntry(
            timestamp=time.time(),
            session_id=self.get_session_id(),
            component=component,
            operation=operation,
            duration_ms=duration_ms,
            memory_mb=memory_mb,
            cpu_percent=cpu_percent,
            gpu_utilization=gpu_utilization
        )

        self._write_structured_log(self.performance_log_file, entry)

        # Update performance stats
        key = f"{component}.{operation}"
        if key not in self._performance_stats:
            self._performance_stats[key] = []
        self._performance_stats[key].append(duration_ms)

        # Log slow operations
        if duration_ms > 1000:  # > 1 second
            self.logger.warning(f"Slow operation: {component}.{operation} took {duration_ms:.1f}ms")

    @contextmanager
    def performance_timer(self, component: str, operation: str):
        """Context manager for timing operations"""
        start_time = time.time()
        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.log_performance(component, operation, duration_ms)

    def _write_structured_log(self, file_path: Path, entry):
        """Write structured log entry to JSONL file"""
        try:
            # Ensure parent directory exists and create file lazily
            file_path.parent.mkdir(exist_ok=True, parents=True)
            with open(file_path, 'a') as f:
                json.dump(asdict(entry), f, default=str)
                f.write('\n')
        except Exception as e:
            self.logger.error(f"Failed to write structured log: {e}")

    def get_session_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get summary statistics for a session"""
        if session_id is None:
            session_id = self.get_session_id()

        summary = {
            'session_id': session_id,
            'emotions': self._analyze_emotion_logs(session_id),
            'music': self._analyze_music_logs(session_id),
            'feedback': self._analyze_feedback_logs(session_id),
            'performance': self._analyze_performance_logs(session_id)
        }

        return summary

    def _analyze_emotion_logs(self, session_id: str) -> Dict[str, Any]:
        """Analyze emotion logs for a session"""
        emotions = self._read_structured_logs(self.emotion_log_file, session_id)

        if not emotions:
            return {'count': 0}

        valences = [e['valence'] for e in emotions]
        arousals = [e['arousal'] for e in emotions]
        confidences = [e['confidence'] for e in emotions]

        return {
            'count': len(emotions),
            'duration_minutes': (emotions[-1]['timestamp'] - emotions[0]['timestamp']) / 60,
            'valence': {
                'mean': float(np.mean(valences)),
                'std': float(np.std(valences)),
                'min': float(np.min(valences)),
                'max': float(np.max(valences))
            },
            'arousal': {
                'mean': float(np.mean(arousals)),
                'std': float(np.std(arousals)),
                'min': float(np.min(arousals)),
                'max': float(np.max(arousals))
            },
            'confidence': {
                'mean': float(np.mean(confidences)),
                'min': float(np.min(confidences))
            },
            'sources': {src: sum(1 for e in emotions if e['source'] == src)
                       for src in set(e['source'] for e in emotions)}
        }

    def _analyze_music_logs(self, session_id: str) -> Dict[str, Any]:
        """Analyze music generation logs for a session"""
        music_logs = self._read_structured_logs(self.music_log_file, session_id)

        if not music_logs:
            return {'count': 0}

        generation_times = [m['generation_time_ms'] for m in music_logs
                           if m.get('generation_time_ms') is not None]
        dtw_errors = [m['dtw_error'] for m in music_logs
                     if m.get('dtw_error') is not None]

        trajectories = [m['trajectory_type'] for m in music_logs
                       if m.get('trajectory_type') is not None]

        analysis = {
            'count': len(music_logs),
            'trajectories': {traj: trajectories.count(traj) for traj in set(trajectories)} if trajectories else {},
        }

        if generation_times:
            analysis['generation_time_ms'] = {
                'mean': float(np.mean(generation_times)),
                'std': float(np.std(generation_times)),
                'max': float(np.max(generation_times))
            }

        if dtw_errors:
            analysis['dtw_error'] = {
                'mean': float(np.mean(dtw_errors)),
                'std': float(np.std(dtw_errors)),
                'trend': 'improving' if dtw_errors[-1] < dtw_errors[0] else 'worsening'
            }

        return analysis

    def _analyze_feedback_logs(self, session_id: str) -> Dict[str, Any]:
        """Analyze feedback logs for a session"""
        feedback_logs = self._read_structured_logs(self.feedback_log_file, session_id)

        if not feedback_logs:
            return {'count': 0}

        ratings = [f['rating'] for f in feedback_logs
                  if f.get('rating') is not None]
        categories = [f['category'] for f in feedback_logs
                     if f.get('category') is not None]

        analysis = {
            'count': len(feedback_logs),
            'categories': {cat: categories.count(cat) for cat in set(categories)} if categories else {},
            'feedback_types': {ft: sum(1 for f in feedback_logs if f['feedback_type'] == ft)
                              for ft in set(f['feedback_type'] for f in feedback_logs)}
        }

        if ratings:
            analysis['ratings'] = {
                'mean': float(np.mean(ratings)),
                'std': float(np.std(ratings)),
                'count': len(ratings)
            }

        return analysis

    def _analyze_performance_logs(self, session_id: str) -> Dict[str, Any]:
        """Analyze performance logs for a session"""
        perf_logs = self._read_structured_logs(self.performance_log_file, session_id)

        if not perf_logs:
            return {'count': 0}

        # Group by component.operation
        operations = {}
        for log in perf_logs:
            key = f"{log['component']}.{log['operation']}"
            if key not in operations:
                operations[key] = []
            operations[key].append(log['duration_ms'])

        # Analyze each operation
        operation_stats = {}
        for op, durations in operations.items():
            operation_stats[op] = {
                'count': len(durations),
                'mean_ms': float(np.mean(durations)),
                'max_ms': float(np.max(durations)),
                'total_ms': float(np.sum(durations))
            }

        return {
            'count': len(perf_logs),
            'operations': operation_stats,
            'total_processing_time_ms': float(sum(log['duration_ms'] for log in perf_logs))
        }

    def _read_structured_logs(self, file_path: Path, session_id: str) -> List[Dict]:
        """Read structured logs for a specific session"""
        logs = []

        try:
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if data.get('session_id') == session_id:
                            logs.append(data)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass

        return logs

    def get_performance_report(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get system performance report for the last N hours"""
        cutoff_time = time.time() - (hours_back * 3600)

        # Read recent performance logs
        recent_logs = []
        try:
            with open(self.performance_log_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if data['timestamp'] >= cutoff_time:
                            recent_logs.append(data)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            return {'error': 'No performance data available'}

        if not recent_logs:
            return {'error': 'No recent performance data'}

        # Analyze by component
        component_stats = {}
        for log in recent_logs:
            component = log['component']
            if component not in component_stats:
                component_stats[component] = {
                    'operations': {},
                    'total_time_ms': 0,
                    'call_count': 0
                }

            operation = log['operation']
            duration = log['duration_ms']

            if operation not in component_stats[component]['operations']:
                component_stats[component]['operations'][operation] = []

            component_stats[component]['operations'][operation].append(duration)
            component_stats[component]['total_time_ms'] += duration
            component_stats[component]['call_count'] += 1

        # Calculate statistics
        for component, stats in component_stats.items():
            for operation, durations in stats['operations'].items():
                stats['operations'][operation] = {
                    'count': len(durations),
                    'mean_ms': float(np.mean(durations)),
                    'std_ms': float(np.std(durations)),
                    'max_ms': float(np.max(durations)),
                    'min_ms': float(np.min(durations)),
                    'total_ms': float(np.sum(durations))
                }

        return {
            'time_window_hours': hours_back,
            'total_operations': len(recent_logs),
            'components': component_stats,
            'system_stats': {
                'total_processing_time_ms': sum(log['duration_ms'] for log in recent_logs),
                'unique_sessions': len(set(log['session_id'] for log in recent_logs))
            }
        }

    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up log files older than specified days"""
        cutoff_time = time.time() - (days_to_keep * 24 * 3600)

        for log_file in [self.emotion_log_file, self.music_log_file,
                        self.feedback_log_file, self.performance_log_file]:
            if not log_file.exists():
                continue

            # Read and filter logs
            kept_logs = []
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            if data['timestamp'] >= cutoff_time:
                                kept_logs.append(line.strip())
                        except json.JSONDecodeError:
                            continue

                # Rewrite file with kept logs
                with open(log_file, 'w') as f:
                    for log_line in kept_logs:
                        f.write(log_line + '\n')

                self.logger.info(f"Cleaned {log_file.name}: kept {len(kept_logs)} entries")

            except Exception as e:
                self.logger.error(f"Failed to clean {log_file.name}: {e}")

    def export_session_data(self, session_id: str, output_file: Optional[Path] = None) -> Path:
        """Export all data for a session to a single JSON file"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.log_dir / f"session_{session_id}_{timestamp}.json"

        session_data = {
            'session_id': session_id,
            'export_timestamp': time.time(),
            'summary': self.get_session_summary(session_id),
            'raw_data': {
                'emotions': self._read_structured_logs(self.emotion_log_file, session_id),
                'music': self._read_structured_logs(self.music_log_file, session_id),
                'feedback': self._read_structured_logs(self.feedback_log_file, session_id),
                'performance': self._read_structured_logs(self.performance_log_file, session_id)
            }
        }

        with open(output_file, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)

        self.logger.info(f"Session data exported to {output_file}")
        return output_file

    def info(self, message, *args, **kwargs):
        """Forward info calls to internal logger"""
        return self.logger.info(message, *args, **kwargs)
    
    def error(self, message, *args, **kwargs):
        """Forward error calls to internal logger"""
        return self.logger.error(message, *args, **kwargs)
    
    def warning(self, message, *args, **kwargs):
        """Forward warning calls to internal logger"""
        return self.logger.warning(message, *args, **kwargs)
    
    def debug(self, message, *args, **kwargs):
        """Forward debug calls to internal logger (FIXED: now uses debug level)"""
        return self.logger.debug(message, *args, **kwargs)
    
    def exception(self, message, *args, **kwargs):
        """Forward exception calls to internal logger"""
        return self.logger.exception(message, *args, **kwargs)
    
    def critical(self, message, *args, **kwargs):
        """Forward critical calls to internal logger"""
        return self.logger.critical(message, *args, **kwargs)


# Global logger instance
_logger_instance = None


def get_logger() -> EmoTuneLogger:
    """Get the global EmoTune logger instance. Always use this in codebase, not logging.info() directly."""
    global _logger_instance
    if _logger_instance is None:
        # Default to DEBUG for both logger and file handler
        _logger_instance = EmoTuneLogger(log_dir="logs", log_level=logging.DEBUG)
    return _logger_instance

def init_logger(log_dir: str = "logs", log_level: int = logging.INFO) -> EmoTuneLogger:
    """Initialize the global EmoTune logger"""
    global _logger_instance
    _logger_instance = EmoTuneLogger(log_dir=log_dir, log_level=log_level)
    return _logger_instance

def setup_logging(log_dir: str = "logs", log_level: int = logging.INFO):
    """Setup logging system and return configured logger"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = EmoTuneLogger(log_dir=log_dir, log_level=log_level)
    return _logger_instance

# Convenience functions
def log_emotion(valence: float, arousal: float, confidence: float, source: str, **kwargs):
    """Convenience function to log emotion data"""
    get_logger().log_emotion(valence, arousal, confidence, source, **kwargs)

def log_music(parameters: Dict[str, float], **kwargs):
    """Convenience function to log music generation"""
    get_logger().log_music_generation(parameters, **kwargs)

def log_feedback(feedback_type: str, **kwargs):
    """Convenience function to log user feedback"""
    get_logger().log_feedback(feedback_type, **kwargs)

def performance_timer(component: str, operation: str):
    """Convenience function for performance timing"""
    return get_logger().performance_timer(component, operation)

if __name__ == "__main__":
    # Example usage and testing
    logger = EmoTuneLogger(log_dir="test_logs")
    logger.set_session_id("test_session_001")

    # Test emotion logging
    logger.log_emotion(0.5, -0.2, 0.85, "face", raw_data={"landmarks": [1, 2, 3]})
    logger.log_emotion(0.3, -0.1, 0.90, "voice")

    # Test music logging
    logger.log_music_generation(
        {"tempo": 120, "valence": 0.5},
        trajectory_type="calm_down",
        dtw_error=0.12,
        generation_time_ms=45.2
    )

    # Test feedback logging
    logger.log_feedback("explicit", rating=4.5, category="like")

    # Test performance timing
    with logger.performance_timer("emotion_analysis", "face_detection"):
        time.sleep(0.1)  # Simulate work

    # Get session summary
    summary = logger.get_session_summary()
    print("Session Summary:")
    print(json.dumps(summary, indent=2, default=str))

    # Export session data
    export_file = logger.export_session_data("test_session_001")
    print(f"Session data exported to: {export_file}")

# NOTE: Always use get_logger() and logger.info()/debug()/error() in codebase, not logging.info() directly.
# This ensures all logs go through the custom EmoTuneLogger and are properly structured/handled.
