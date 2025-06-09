"""Demonstration of DTW-based adaptive trajectory planning for EmoTune."""
import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw
from emotune.core.trajectory.dtw_matcher import DTWMatcher  # Assuming this exists based on project structure

def generate_trajectory(length=100, start=(0, 0), end=(1, 1), noise=0.1):
    """Generate synthetic valence/arousal trajectory with noise."""
    t = np.linspace(0, 1, length)
    x = start[0] + (end[0] - start[0]) * t + np.random.normal(0, noise, length)
    y = start[1] + (end[1] - start[1]) * t + np.random.normal(0, noise, length)
    return np.column_stack((x, y))

def plot_trajectories(ax, target, actual, title=""):
    """Plot valence/arousal trajectories."""
    ax.plot(target[:, 0], target[:, 1], 'b-', label='Target')
    ax.plot(actual[:, 0], actual[:, 1], 'r-', label='Actual')
    ax.set_xlabel('Valence')
    ax.set_ylabel('Arousal')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

def main():
    # Generate sample trajectories
    target_traj = generate_trajectory(50, (0.2, 0.8), (0.8, 0.2), 0.05)
    actual_traj = generate_trajectory(70, (0.1, 0.7), (0.9, 0.3), 0.1)

    # Initialize DTW matcher
    matcher = DTWMatcher()
    distance, path = matcher.match(target_traj, actual_traj)

    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Plot original trajectories
    plot_trajectories(ax1, target_traj, actual_traj, "Original Trajectories")

    # Plot DTW alignment path
    ax2.plot(path[0], path[1], 'k-', alpha=0.5)
    ax2.set_xlabel('Target Index')
    ax2.set_ylabel('Actual Index')
    ax2.set_title(f'DTW Alignment Path (distance={distance:.2f})')
    ax2.grid(True)

    # Plot warped trajectories
    warped_actual = actual_traj[path[1]]
    plot_trajectories(ax3, target_traj, warped_actual, "Warped Trajectories")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
