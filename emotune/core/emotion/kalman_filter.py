import numpy as np
from typing import Dict, Tuple
import logging

from emotune.utils.logging import get_logger
logger = get_logger()

class KalmanEmotionFilter:
    """Kalman filter for probabilistic emotion smoothing"""
    
    def __init__(self, dt: float = 1.0, process_noise: float = 0.01, 
                 measurement_noise: float = 0.1):
        self.dt = dt
        
        # State: [valence, arousal, d_valence/dt, d_arousal/dt]
        self.state_dim = 4
        self.obs_dim = 2
        
        # Initialize state
        self.x = np.zeros(self.state_dim)  # State vector
        self.P = np.eye(self.state_dim) * 0.5  # Covariance matrix
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Observation matrix (observe position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise covariance
        self.Q = np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2]
        ]) * process_noise
        
        # Default measurement noise
        self.R_default = np.eye(self.obs_dim) * measurement_noise
        
        self.initialized = False
        
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prediction step"""
        # Predict state
        self.x = self.F @ self.x
        
        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x.copy(), self.P.copy()
    
    def update(self, observation: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Update step with emotion observation, with validity checks and adaptive covariance."""
        v = observation.get('valence', 0.0)
        a = observation.get('arousal', 0.0)
        confidence = observation.get('confidence', 0.5)
        
        # FIXED: Handle numpy arrays properly
        if isinstance(v, np.ndarray):
            v = float(v.item() if v.size == 1 else v[0])
        if isinstance(a, np.ndarray):
            a = float(a.item() if a.size == 1 else a[0])
        
        z = np.array([v, a])
        
        # Adaptive measurement noise based on uncertainty and confidence
        uncertainty = observation.get('uncertainty', 0.5)
        conf = float(confidence) if confidence is not None else 0.5
        conf = np.clip(conf, 0.0, 1.0)
        R = self.R_default * (1.0 + uncertainty * 2.0) * (1.0 + (1.0 - conf) * 2.0)
        
        # Validity checks - FIXED: Use proper numpy methods instead of boolean array comparison
        if not (np.isfinite(v) and np.isfinite(a)):
            logger.warning("Kalman update received non-finite valence/arousal. Skipping update.")
            return self.x.copy(), self.P.copy()
        
        if not self.initialized:
            # Initialize state with first observation
            self.x[:2] = z
            self.x[2:] = 0  # Zero velocity
            self.initialized = True
            return self.x.copy(), self.P.copy()
        
        # Innovation
        y = z - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        I_KH = np.eye(self.state_dim) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        
        return self.x.copy(), self.P.copy()
    
    def get_emotion_distribution(self) -> Dict:
        """Get current emotion estimate as Gaussian distribution"""
        # Extract position and covariance
        mu = self.x[:2]  # [valence, arousal]
        Sigma = self.P[:2, :2]  # 2x2 covariance matrix
        
        # Calculate uncertainty measures
        trace_cov = np.trace(Sigma)
        det_cov = np.linalg.det(Sigma)
        
        return {
            'mean': {
                'valence': float(mu[0]),
                'arousal': float(mu[1])
            },
            'covariance': Sigma.tolist(),
            'velocity': {
                'valence': float(self.x[2]),
                'arousal': float(self.x[3])
            },
            'uncertainty_trace': float(trace_cov),
            'uncertainty_det': float(det_cov)
        }
    
    def reset(self):
        """Reset filter state"""
        self.x = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim) * 0.5
        self.initialized = False

