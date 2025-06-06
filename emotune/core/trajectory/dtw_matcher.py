import numpy as np
from typing import List, Tuple, Dict
import logging

from emotune.utils.logging import get_logger
logger = get_logger()

class DTWMatcher:
    """Dynamic Time Warping for trajectory matching"""
    
    def __init__(self, window_size: int = None):
        self.window_size = window_size
        
    def compute_dtw_distance(self, trajectory1: List[Tuple[float, float]], 
                           trajectory2: List[Tuple[float, float]]) -> float:
        """
        Compute DTW distance between two trajectories
        Each trajectory is a list of (valence, arousal) tuples
        """
        if not trajectory1 or not trajectory2:
            return float('inf')
            
        n, m = len(trajectory1), len(trajectory2)
        
        # Create distance matrix
        dist_matrix = np.full((n, m), np.inf)
        
        # Compute pairwise distances
        for i in range(n):
            for j in range(m):
                if self._in_window(i, j, n, m):
                    dist_matrix[i, j] = self._euclidean_distance(
                        trajectory1[i], trajectory2[j]
                    )
        
        # DTW dynamic programming
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if self._in_window(i-1, j-1, n, m):
                    cost = dist_matrix[i-1, j-1]
                    dtw_matrix[i, j] = cost + min(
                        dtw_matrix[i-1, j],      # insertion
                        dtw_matrix[i, j-1],      # deletion
                        dtw_matrix[i-1, j-1]     # match
                    )
        
        return dtw_matrix[n, m]
    
    def compute_trajectory_deviation(self, actual_trajectory: List[Dict],
                                   target_trajectory_fn: callable,
                                   start_time: float) -> float:
        """
        Compute deviation of actual trajectory from target
        """
        if not actual_trajectory:
            return 1.0  # Maximum deviation
            
        # Extract actual path
        actual_path = [
            (point['mean']['valence'], point['mean']['arousal'])
            for point in actual_trajectory
        ]
        
        # Generate target path at corresponding times
        target_path = []
        for point in actual_trajectory:
            t = point['timestamp'] - start_time
            val, aro = target_trajectory_fn(t)
            target_path.append((val, aro))
        
        # Compute DTW distance
        dtw_dist = self.compute_dtw_distance(actual_path, target_path)
        
        # Normalize by trajectory length and scale
        max_possible_dist = len(actual_path) * 2.0  # Max distance in VA space
        normalized_deviation = min(1.0, dtw_dist / max_possible_dist)
        
        return normalized_deviation
    
    def _euclidean_distance(self, point1: Tuple[float, float], 
                           point2: Tuple[float, float]) -> float:
        """Compute Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _in_window(self, i: int, j: int, n: int, m: int) -> bool:
        """Check if (i,j) is within the DTW window constraint"""
        if self.window_size is None:
            return True
            
        # Sakoe-Chiba band
        return abs(i - j * n / m) <= self.window_size
