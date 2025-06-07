# EmoTune: AI-Powered Adaptive Music Therapy

**EmoTune** is a real-time, single-user, emotion-responsive system that combines facial and vocal emotion analysis with adaptive music generation. It explores AI-driven music therapy for mental well-being by dynamically responding to users' affective states.

---

## Overview

EmoTune captures and analyzes emotional signals through facial expressions and vocal tone, interprets the user's current affective state (valence, arousal, and discrete emotion), and adapts music output in real time. The system is modular, with independent perception, feedback, and synthesis components.

---

## Features

- **Multimodal Emotion Analysis**: Real-time facial and vocal emotion recognition using deep learning (EmoNet, MTCNN, etc.).
- **Valence-Arousal Mapping**: Continuous affective state estimation and trajectory tracking.
- **Adaptive Music Generation**: Dynamic music parameterization and synthesis via FoxDot, Sonic Pi, MIDI, or TidalCycles engines.
- **Reinforcement Learning**: Soft Actor-Critic (SAC) agent adapts music parameters to optimize user affect and feedback.
- **User Feedback Integration**: Collects explicit (ratings, comfort, effectiveness, comments) and implicit feedback for RL.
- **Web Dashboard**: Interactive frontend for session control, emotion visualization, trajectory tracking, and feedback.
- **Session Logging**: Stores emotion, music, and feedback data for analysis and research.

---

## Architecture

- **Backend**: Python (Flask, Flask-SocketIO), modular core (emotion, music, trajectory, feedback, RL), threaded session manager.
- **Frontend**: HTML/JS dashboard 
- **RL Policy**: Soft Actor-Critic (SAC) with a two-layer neural network policy 
- **Emotion Models**: Uses vendor/emonet for facial emotion, MTCNN for face detection, and Wav2Vec2 for voice.

---

## Installation

1. **Clone the repository** (with submodules):
   ```powershell
   git clone --recurse-submodules <repo-url>
   cd emotune
   ```
2. **Install dependencies** (Python 3.9+ recommended):
   ```powershell
   pip install -r requirements.txt
   ```
3. **Download/prep emotion model weights** (see `vendor/emonet/pretrained/`).

---

## Usage

1. **Start the server:**
   ```powershell
   python -m emotune
   ```
2. **Open the dashboard:**
   - Visit [http://localhost:5000](http://localhost:5000) in your browser.
3. **Start a session:**
   - Select a trajectory and duration, then click "Start Session".
   - Allow camera/microphone access for emotion capture.
   - View real-time emotion, music, and trajectory feedback.
   - Submit feedback at session end.

---

## Configuration

- Edit `config/default.json` for system/session parameters.
- Music and trajectory templates: `emotune/config/music_params.py`, `emotune/config/trajectories.py`.

---

## Project Structure

- `emotune/core/` — Main logic (emotion, music, RL, feedback, trajectory)
- `emotune/web/` — Flask app, dashboard, static assets
- `vendor/emonet/` — Facial emotion model (submodule)
- `logs/` — Session logs (emotion, music, feedback, performance)
- `tests/` — Example tests

---

## Research & Extensibility

- Modular for research: swap models, add new feedback types, extend RL, or integrate new music engines.
- Feedback and emotion logs support offline analysis.
- RL reward and state logic can be customized for new experiments.

---

## Requirements

- Python 3.9+
- Camera and microphone for emotion capture


## Acknowledgments

- [EmoNet](https://github.com/face-analysis/emonet) for facial emotion recognition
- [FoxDot](https://foxdot.org/), [Sonic Pi](https://sonic-pi.net/), [TidalCycles](https://tidalcycles.org/) for music engines
