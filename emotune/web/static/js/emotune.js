/**
 * EmoTune Main JavaScript - Client-side logic for real-time emotion monitoring and music control
 */

class EmoTuneClient {
    constructor() {
        this.socket = null;
        this.isConnected = false;
        this.currentSession = null;
        this.emotionMonitoring = false;
        this.audioContext = null;
        this.trajectoryCanvas = null;
        this.trajectoryCtx = null;
        
        this.init();
    }
    
    init() {
        this.setupSocketConnection();
        this.setupEventListeners();
        this.setupCanvas();
        this.setupMediaCapture();
        this.updateUI();
    }
    
    setupSocketConnection() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            console.log('Connected to EmoTune server');
            this.isConnected = true;
            this.updateConnectionStatus(true);
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from EmoTune server');
            this.isConnected = false;
            this.updateConnectionStatus(false);
        });
        
        this.socket.on('emotion_update', (data) => {
            this.handleEmotionUpdate(data);
        });
        
        this.socket.on('monitoring_started', () => {
            console.log('Emotion monitoring started');
            this.emotionMonitoring = true;
        });
        
        this.socket.on('music_status', (data) => {
            this.showNotification(`Music status: ${data.status}`, 'info');
        });
        this.socket.on('music_parameters', (params) => {
            this.updateMusicParameters(params);
        });
        this.socket.on('trajectory_info', (info) => {
            // Optionally update trajectory visualization or info panel
        });
        this.socket.on('session_status', (data) => {
            if (data.active) {
                this.currentSession = {
                    id: data.session_id,
                    trajectory_type: data.trajectory_type,
                    duration: data.duration,
                    startTime: Date.now()
                };
                this.updateSessionUI(true);
            } else {
                this.currentSession = null;
                this.updateSessionUI(false);
            }
        });
        this.socket.on('trajectory_progress', (progress) => {
            // Update trajectory visualization and info fields
            this.updateTrajectoryVisualization(progress);
            if (progress.info && progress.info.target) {
                document.getElementById('targetEmotion').textContent =
                    `(${progress.info.target.valence.toFixed(2)}, ${progress.info.target.arousal.toFixed(2)})`;
            }
            if (progress.deviation !== undefined) {
                document.getElementById('trajectoryDeviation').textContent =
                    progress.deviation.toFixed(3);
            }
        });
        this.socket.on('error', (data) => {
            this.showNotification(data.message, 'error');
        });
    }
    
    setupEventListeners() {
        // Session controls
        document.getElementById('startSessionBtn').addEventListener('click', () => this.startSession());
        document.getElementById('stopSessionBtn').addEventListener('click', () => this.stopSession());
        
        // Duration slider
        const durationSlider = document.getElementById('durationSlider');
        durationSlider.addEventListener('input', (e) => {
            document.getElementById('durationValue').textContent = e.target.value;
        });
        
        // Music controls
        document.getElementById('playMusicBtn').addEventListener('click', () => this.playMusic());
        document.getElementById('pauseMusicBtn').addEventListener('click', () => this.pauseMusic());
        document.getElementById('generateMusicBtn').addEventListener('click', () => this.generateMusic());
        
        // Feedback controls
        document.querySelectorAll('.rating-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.selectRating(e.target));
        });
        
        const comfortSlider = document.getElementById('comfortSlider');
        comfortSlider.addEventListener('input', (e) => {
            document.getElementById('comfortValue').textContent = e.target.value;
        });
        
        const effectivenessSlider = document.getElementById('effectivenessSlider');
        effectivenessSlider.addEventListener('input', (e) => {
            document.getElementById('effectivenessValue').textContent = e.target.value;
        });
        
        document.getElementById('submitFeedbackBtn').addEventListener('click', () => this.submitFeedback());
    }
    
    setupCanvas() {
        this.trajectoryCanvas = document.getElementById('trajectoryCanvas');
        this.trajectoryCtx = this.trajectoryCanvas.getContext('2d');
        this.drawTrajectoryGrid();
    }
    
    async setupMediaCapture() {
        // DO NOT open the camera or microphone in the browser!
        // All camera/audio capture is handled by the backend (Python) only.
        // This prevents resource conflicts and ensures only one process controls the webcam.
        // If you want a preview, implement a backend-to-frontend streaming solution instead.
        // This function is now a no-op for safety.
        console.log('[EmoTune] setupMediaCapture: Camera/mic access is disabled in frontend to avoid resource conflicts.');
    }
    
    updateConnectionStatus(connected) {
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        
        if (connected) {
            statusDot.classList.add('connected');
            statusText.textContent = 'Connected';
        } else {
            statusDot.classList.remove('connected');
            statusText.textContent = 'Disconnected';
        }
    }
    
    async startSession() {
        if (!this.isConnected) {
            this.showNotification('Not connected to server', 'error');
            return;
        }
        
        const trajectoryType = document.getElementById('trajectorySelect').value;
        const duration = parseInt(document.getElementById('durationSlider').value) * 60; // Convert to seconds
        
        try {
            const response = await fetch('/session/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ trajectory_type: trajectoryType, duration: duration })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.currentSession = {
                    id: data.session_id,
                    trajectory_type: trajectoryType,
                    duration: duration,
                    startTime: Date.now()
                };
                
                this.updateSessionUI(true);
                this.startEmotionMonitoring();
                this.showNotification('Session started successfully', 'success');
            } else {
                this.showNotification(data.error || 'Failed to start session', 'error');
            }
        } catch (error) {
            console.error('Error starting session:', error);
            this.showNotification('Failed to start session', 'error');
        }
    }
    
    async stopSession() {
        if (!this.currentSession) return;
        
        try {
            const response = await fetch('/session/stop', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.currentSession = null;
                this.emotionMonitoring = false;
                this.updateSessionUI(false);
                this.showNotification('Session stopped successfully', 'success');
            } else {
                this.showNotification(data.error || 'Failed to stop session', 'error');
            }
        } catch (error) {
            console.error('Error stopping session:', error);
            this.showNotification('Failed to stop session', 'error');
        }
    }
    
    startEmotionMonitoring() {
        if (!this.currentSession) return;
        this.socket.emit('start_emotion_monitoring');
        // No need to start captureEmotionData interval; backend handles all capture and emission.
        // Only update session progress bar locally.
        this.progressInterval = setInterval(() => {
            this.updateSessionProgress();
        }, 1000);
    }

    handleEmotionUpdate(data) {
        // Update emotion display
        if (data.emotion_state) this.updateEmotionDisplay(data.emotion_state);
        // Update trajectory visualization and info
        if (data.trajectory_progress) {
            this.updateTrajectoryVisualization(data.trajectory_progress);
            // Robustly update target and deviation fields
            let info = data.trajectory_progress.info || {};
            if (info.target && typeof info.target.valence === 'number' && typeof info.target.arousal === 'number') {
                document.getElementById('targetEmotion').textContent =
                    `(${info.target.valence.toFixed(2)}, ${info.target.arousal.toFixed(2)})`;
            } else {
                document.getElementById('targetEmotion').textContent = '-';
            }
            if (typeof data.trajectory_progress.deviation === 'number') {
                document.getElementById('trajectoryDeviation').textContent =
                    data.trajectory_progress.deviation.toFixed(3);
            } else {
                document.getElementById('trajectoryDeviation').textContent = '-';
            }
        }
        // Update music parameters display
        if (data.music_parameters) this.updateMusicParameters(data.music_parameters);
        // RL/adaptation/feedback status display
        if (data.rl_status) {
            const rlDiv = document.getElementById('rlStatus');
            if (rlDiv) {
                let html = '';
                if (data.rl_status.training_summary) {
                    html += `<b>RL Buffer:</b> ${data.rl_status.training_summary.buffer_size} <br/>`;
                    html += `<b>Current Params:</b> ${JSON.stringify(data.rl_status.training_summary.current_params)} <br/>`;
                }
                if (data.rl_status.feedback) {
                    html += `<b>Feedback Reward:</b> ${data.rl_status.feedback.reward?.toFixed(3)} <br/>`;
                    html += `<b>Feedback Confidence:</b> ${data.rl_status.feedback.confidence?.toFixed(2)} <br/>`;
                    html += `<b>Feedback Trend:</b> ${data.rl_status.feedback.trend?.toFixed(3)} <br/>`;
                }
                if (data.rl_status.adaptation) {
                    html += `<b>Adaptations:</b> ${data.rl_status.adaptation.total_adaptations} <br/>`;
                    html += `<b>Recent Adaptations:</b> ${data.rl_status.adaptation.recent_adaptations} <br/>`;
                    html += `<b>Adaptation Rate:</b> ${data.rl_status.adaptation.adaptation_rate?.toFixed(3)} /min <br/>`;
                }
                rlDiv.innerHTML = html;
            }
        }
    }

    updateEmotionDisplay(emotionState) {
        const valence = emotionState.valence || 0;
        const arousal = emotionState.arousal || 0;
        const confidence = emotionState.confidence || 0;
        
        // Update emotion circle position
        const emotionPoint = document.getElementById('emotionPoint');
        const circleRadius = 90; // Half of circle width minus point radius
        
        const x = (valence * circleRadius) + circleRadius;
        const y = (-arousal * circleRadius) + circleRadius;
        
        emotionPoint.style.left = `${x}px`;
        emotionPoint.style.top = `${y}px`;
        
        // Update values display
        document.getElementById('valenceValue').textContent = valence.toFixed(2);
        document.getElementById('arousalValue').textContent = arousal.toFixed(2);
        document.getElementById('confidenceValue').textContent = `${(confidence * 100).toFixed(0)}%`;
    }
    
    updateTrajectoryVisualization(trajectoryProgress) {
        if (!this.trajectoryCtx || !trajectoryProgress) return;
        const ctx = this.trajectoryCtx;
        const canvas = this.trajectoryCanvas;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        this.drawTrajectoryGrid();
        // Draw target trajectory
        let targetPath = trajectoryProgress.target_path || (trajectoryProgress.info && trajectoryProgress.info.target_path);
        if (Array.isArray(targetPath) && targetPath.length > 0) {
            ctx.strokeStyle = '#667eea';
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.beginPath();
            targetPath.forEach((point, index) => {
                const x = (point.valence + 1) * canvas.width / 2;
                const y = (1 - point.arousal) * canvas.height / 2;
                if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            });
            ctx.stroke();
        }
        // Draw actual trajectory
        let actualPath = trajectoryProgress.actual_path || (trajectoryProgress.info && trajectoryProgress.info.actual_path);
        if (Array.isArray(actualPath) && actualPath.length > 0) {
            ctx.strokeStyle = '#4ecdc4';
            ctx.lineWidth = 3;
            ctx.setLineDash([]);
            ctx.beginPath();
            actualPath.forEach((point, index) => {
                const x = (point.valence + 1) * canvas.width / 2;
                const y = (1 - point.arousal) * canvas.height / 2;
                if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            });
            ctx.stroke();
        }
        // Update deviation info if present
        if (typeof trajectoryProgress.deviation === 'number') {
            document.getElementById('trajectoryDeviation').textContent = trajectoryProgress.deviation.toFixed(3);
        } else if (trajectoryProgress.info && typeof trajectoryProgress.info.deviation === 'number') {
            document.getElementById('trajectoryDeviation').textContent = trajectoryProgress.info.deviation.toFixed(3);
        }
    }
    
    drawTrajectoryGrid() {
        const ctx = this.trajectoryCtx;
        const canvas = this.trajectoryCanvas;
        
        ctx.strokeStyle = '#e0e0e0';
        ctx.lineWidth = 1;
        
        // Draw grid lines
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        
        // Vertical center line
        ctx.beginPath();
        ctx.moveTo(centerX, 0);
        ctx.lineTo(centerX, canvas.height);
        ctx.stroke();
        
        // Horizontal center line
        ctx.beginPath();
        ctx.moveTo(0, centerY);
        ctx.lineTo(canvas.width, centerY);
        ctx.stroke();
        
        // Add labels
        ctx.fillStyle = '#666';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        
        ctx.fillText('Positive', canvas.width - 30, centerY - 5);
        ctx.fillText('Negative', 30, centerY - 5);
        ctx.fillText('High Arousal', centerX, 15);
        ctx.fillText('Low Arousal', centerX, canvas.height - 5);
    }
    
    updateMusicParameters(musicParams) {
        if (!musicParams) return;
        // Update tempo
        if (musicParams.tempo !== undefined) {
            document.getElementById('tempoValue').textContent = Math.round(musicParams.tempo);
        } else if (musicParams.tempo_bpm !== undefined) {
            document.getElementById('tempoValue').textContent = Math.round(musicParams.tempo_bpm);
        } else {
            document.getElementById('tempoValue').textContent = '-';
        }
        // Update key
        if (musicParams.key) {
            document.getElementById('keyValue').textContent = musicParams.key;
        } else if (musicParams.scale) {
            document.getElementById('keyValue').textContent = musicParams.scale;
        } else {
            document.getElementById('keyValue').textContent = '-';
        }
        // Update parameter bars (handle undefined gracefully)
        this.updateParameterBar('rhythmComplexity', musicParams.rhythm_complexity);
        this.updateParameterBar('harmonicComplexity', musicParams.harmonic_complexity);
        this.updateParameterBar('textureDensity', musicParams.texture_density);
        this.updateParameterBar('volumeLevel', musicParams.volume || musicParams.overall_volume);
    }
    
    updateParameterBar(elementId, value) {
        const element = document.getElementById(elementId);
        if (element && value !== undefined) {
            const percentage = Math.max(0, Math.min(100, value * 100));
            element.style.width = `${percentage}%`;
        }
    }
    
    updateSessionUI(active) {
        const startBtn = document.getElementById('startSessionBtn');
        const stopBtn = document.getElementById('stopSessionBtn');
        const sessionInfo = document.getElementById('sessionInfo');
        
        if (active && this.currentSession) {
            startBtn.disabled = true;
            stopBtn.disabled = false;
            sessionInfo.style.display = 'block';
            
            document.getElementById('currentGoal').textContent = 
                this.currentSession.trajectory_type.replace('_', ' ').toUpperCase();
            document.getElementById('currentDuration').textContent = 
                `${Math.round(this.currentSession.duration / 60)} min`;
        } else {
            startBtn.disabled = false;
            stopBtn.disabled = true;
            sessionInfo.style.display = 'none';
            
            // Clear intervals
            if (this.emotionCaptureInterval) {
                clearInterval(this.emotionCaptureInterval);
            }
            if (this.progressInterval) {
                clearInterval(this.progressInterval);
            }
        }
    }
    
    updateSessionProgress() {
        if (!this.currentSession) return;
        
        const elapsed = (Date.now() - this.currentSession.startTime) / 1000;
        const progress = Math.min(100, (elapsed / this.currentSession.duration) * 100);
        
        document.getElementById('sessionProgress').style.width = `${progress}%`;
        document.getElementById('progressText').textContent = `${Math.round(progress)}%`;
        
        if (progress >= 100) {
            this.stopSession();
        }
    }
    
    selectRating(button) {
        // Remove selection from all rating buttons
        document.querySelectorAll('.rating-btn').forEach(btn => {
            btn.classList.remove('selected');
        });
        
        // Add selection to clicked button
        button.classList.add('selected');
    }
    
    async submitFeedback() {
        const selectedRating = document.querySelector('.rating-btn.selected');
        const comfort = document.getElementById('comfortSlider').value;
        const effectiveness = document.getElementById('effectivenessSlider').value;
        const comments = document.getElementById('feedbackComments').value;
        
        if (!selectedRating) {
            this.showNotification('Please select a rating', 'warning');
            return;
        }
        
        const feedbackData = {
            rating: parseInt(selectedRating.dataset.rating),
            comfort: parseInt(comfort),
            effectiveness: parseInt(effectiveness),
            comments: comments,
            timestamp: Date.now()
        };
        
        try {
            const response = await fetch('/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(feedbackData)
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showNotification('Feedback submitted successfully', 'success');
                this.clearFeedbackForm();
            } else {
                this.showNotification(data.error || 'Failed to submit feedback', 'error');
            }
        } catch (error) {
            console.error('Error submitting feedback:', error);
            this.showNotification('Failed to submit feedback', 'error');
        }
    }
    
    clearFeedbackForm() {
        document.querySelectorAll('.rating-btn').forEach(btn => {
            btn.classList.remove('selected');
        });
        document.getElementById('comfortSlider').value = 5;
        document.getElementById('comfortValue').textContent = '5';
        document.getElementById('effectivenessSlider').value = 5;
        document.getElementById('effectivenessValue').textContent = '5';
        document.getElementById('feedbackComments').value = '';
    }
    
    playMusic() {
        if (!this.isConnected) return;
        this.socket.emit('music_control', { action: 'play' });
        this.showNotification('Music playback started', 'success');
    }

    pauseMusic() {
        if (!this.isConnected) return;
        this.socket.emit('music_control', { action: 'pause' });
        this.showNotification('Music paused', 'info');
    }

    generateMusic() {
        if (!this.isConnected) return;
        this.socket.emit('music_control', { action: 'regenerate' });
        this.showNotification('Generating new music...', 'info');
    }

    requestMusicParameters() {
        if (!this.isConnected) return;
        this.socket.emit('request_music_parameters');
    }

    requestTrajectoryInfo() {
        if (!this.isConnected) return;
        this.socket.emit('request_trajectory_info');
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        
        // Style the notification
        Object.assign(notification.style, {
            position: 'fixed',
            top: '20px',
            right: '20px',
            padding: '15px 20px',
            borderRadius: '8px',
            color: 'white',
            fontWeight: '600',
            zIndex: '9999',
            transform: 'translateX(100%)',
            transition: 'transform 0.3s ease'
        });
        
        // Set background color based on type
        const colors = {
            success: '#4ecdc4',
            error: '#ff6b6b',
            warning: '#ffe66d',
            info: '#667eea'
        };
        notification.style.backgroundColor = colors[type] || colors.info;
        
        // Add to DOM and animate in
        document.body.appendChild(notification);
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 100);
        
        // Remove after 3 seconds
        setTimeout(() => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }
    
    updateUI() {
        // Initial UI state
        this.updateSessionUI(false);
        this.updateConnectionStatus(false);
    }
}

// Initialize EmoTune client when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.emotuneClient = new EmoTuneClient();
});
