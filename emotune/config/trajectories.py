from core.trajectory.planner import TrajectoryType
from typing import Callable, Dict, Tuple
import numpy as np

def linear_trajectory(start: Tuple[float, float], end: Tuple[float, float], duration: float) -> Callable[[float], Tuple[float, float]]:
    def f(t: float) -> Tuple[float, float]:
        alpha = min(max(t / duration, 0.0), 1.0)
        return (
            start[0] + alpha * (end[0] - start[0]),
            start[1] + alpha * (end[1] - start[1]),
        )
    return f

# Trajectory definitions per type
TRAJECTORY_TEMPLATES: Dict[TrajectoryType, Callable[[float, Dict, Dict], Callable[[float], Tuple[float, float]]]] = {
    TrajectoryType.CALM_DOWN: lambda duration, start, end: linear_trajectory(
        start or (0.5, 0.5), end or (0.8, 0.2), duration
    ),
    TrajectoryType.ENERGIZE: lambda duration, start, end: linear_trajectory(
        start or (0.5, 0.5), end or (0.6, 0.9), duration
    ),
    TrajectoryType.FOCUS: lambda duration, start, end: linear_trajectory(
        start or (0.5, 0.5), end or (0.3, 0.5), duration
    ),
    TrajectoryType.RELAX: lambda duration, start, end: linear_trajectory(
        start or (0.6, 0.5), end or (0.9, 0.1), duration
    ),
    TrajectoryType.MOOD_LIFT: lambda duration, start, end: linear_trajectory(
        start or (0.3, 0.3), end or (0.8, 0.7), duration
    ),
    TrajectoryType.STABILIZE: lambda duration, start, end: linear_trajectory(
        start or (0.4, 0.6), end or (0.5, 0.5), duration
    ),
}


