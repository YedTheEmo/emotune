/**
 * EmoTune Gothic JavaScript - Client-side logic for real-time emotion monitoring and music control
 */

class EmoTuneGothicClient {
    constructor() {
        this.socket = null;
        this.isConnected = false;
        this.currentSession = null;
        this.emotionMonitoring = false;
        this.audioContext = null;
        this.trajectoryCanvas = null;
        this.trajectoryCtx = null;
        this.cameraStream = null;
        this.audioStream = null;
        this.audioAnalyser = null;
        this.audioDataArray = null;
        this.waveformCanvas = null;
        this.waveformCtx = null;
        this.animationFrame = null;
        
        this.init();
    }
    
    init() {
        this.setupSocketConnection();
        this.setupEventListeners();
        this.setupCanvas();
        this.setupMediaCapture();
        this.setupRLFeedback();
        this.updateUI();
    }
    
    setupSocketConnection() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            console.log('Connected to EmoTune Gothic server');
            this.isConnected = true;
            this.updateConnectionStatus(true);
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from EmoTune Gothic server');
            this.isConnected = false;
            this.updateConnectionStatus(false);
        });
        
        this.socket.on('emotion_update', (data) => {
            this.handleEmotionUpdate(data);
        });
        
        this.socket.on('monitoring_started', (data) => {
            if (data && data.status === 'active') {
                this.emotionMonitoring = true;
                this.updateMediaStatus('active');
                this.showGothicNotification('Media monitoring is active.', 'success');
            }
        });
        
        this.socket.on('music_status', (data) => {
            this.showGothicNotification(`Music status: ${data.status}`, 'info');
        });
        
        this.socket.on('music_parameters', (params) => {
            this.updateMusicParameters(params);
        });
        
        this.socket.on('trajectory_info', (info) => {
            this.updateTrajectoryInfo(info);
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

        this.socket.on('feedback_impact', (data) => {
            if (data && data.message) {
                document.getElementById('feedbackImpactText').textContent = data.message;
                this.showGothicNotification('Feedback processed and applied!', 'success');
            }
        });
        
        this.socket.on('trajectory_progress', (progress) => {
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
        
        this.socket.on('rl_status', (data) => {
            this.updateRLStatus(data);
        });
        
        this.socket.on('error', (data) => {
            this.showGothicNotification(data.message, 'error');
            console.error('EmoTune Gothic error:', data.message);
        });
        
        this.socket.on('connect_error', (error) => {
            console.error('Connection error:', error);
            this.showGothicNotification('Connection failed. Retrying...', 'error');
        });
        
        this.socket.on('reconnect', (attemptNumber) => {
            console.log('Reconnected after', attemptNumber, 'attempts');
            this.showGothicNotification('Reconnected to server', 'success');
        });
    }
    
    setupEventListeners() {
        // Session control
        document.getElementById('startSessionBtn').addEventListener('click', () => {
            this.startSession();
        });
        
        document.getElementById('stopSessionBtn').addEventListener('click', () => {
            this.stopSession();
        });
        
        // Duration slider
        const durationSlider = document.getElementById('durationSlider');
        const durationValue = document.getElementById('durationValue');
        durationSlider.addEventListener('input', (e) => {
            durationValue.textContent = `${e.target.value} min`;
        });
        
        document.getElementById('submitFeedbackBtn').addEventListener('click', () => {
            this.submitFeedback();
        });
        
        // Rating buttons
        document.querySelectorAll('.rating-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.selectRating(e.target);
            });
        });
        
        // Feedback sliders
        const comfortSlider = document.getElementById('comfortSlider');
        const effectivenessSlider = document.getElementById('effectivenessSlider');
        const comfortValue = document.getElementById('comfortValue');
        const effectivenessValue = document.getElementById('effectivenessValue');
        
        comfortSlider.addEventListener('input', (e) => {
            comfortValue.textContent = e.target.value;
        });
        
        effectivenessSlider.addEventListener('input', (e) => {
            effectivenessValue.textContent = e.target.value;
        });

        // Confidence threshold sliders
        const faceConfidenceSlider = document.getElementById('faceConfidenceSlider');
        const voiceConfidenceSlider = document.getElementById('voiceConfidenceSlider');
        const faceConfidenceValue = document.getElementById('faceConfidenceValue');
        const voiceConfidenceValue = document.getElementById('voiceConfidenceValue');

        faceConfidenceSlider.addEventListener('input', (e) => {
            faceConfidenceValue.textContent = parseFloat(e.target.value).toFixed(2);
            this.updateConfidenceThresholds();
        });

        voiceConfidenceSlider.addEventListener('input', (e) => {
            voiceConfidenceValue.textContent = parseFloat(e.target.value).toFixed(2);
            this.updateConfidenceThresholds();
        });

        // Analysis mode radio buttons
        const analysisModeRadios = document.querySelectorAll('input[name="analysisMode"]');
        analysisModeRadios.forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.updateAnalysisMode(e.target.value);
            });
        });
    }

    updateConfidenceThresholds() {
        if (!this.isConnected) return;

        const faceThreshold = parseFloat(document.getElementById('faceConfidenceSlider').value);
        const voiceThreshold = parseFloat(document.getElementById('voiceConfidenceSlider').value);

        this.socket.emit('update_confidence_thresholds', {
            face: faceThreshold,
            voice: voiceThreshold
        });
        this.showGothicNotification(`Confidence thresholds updated: Face=${faceThreshold.toFixed(2)}, Voice=${voiceThreshold.toFixed(2)}`, 'info');
    }

    updateAnalysisMode(mode) {
        this.socket.emit('update_analysis_mode', { mode: mode }, (response) => {
            if (response.success) {
                console.log('Analysis mode updated successfully');
            } else {
                console.error('Failed to update analysis mode');
            }
        });
        console.log('Emitting analysis mode:', mode);
    }
    
    setupCanvas() {
        this.trajectoryCanvas = document.getElementById('trajectoryCanvas');
        this.trajectoryCtx = this.trajectoryCanvas.getContext('2d');
        
        // Set canvas size
        this.trajectoryCanvas.width = this.trajectoryCanvas.offsetWidth;
        this.trajectoryCanvas.height = this.trajectoryCanvas.offsetHeight;
        
        this.drawTrajectoryGrid();
    }
    
    async setupMediaCapture() {
        try {
            // Setup camera feed
            const cameraFeed = document.getElementById('cameraFeed');
            const cameraStatus = document.getElementById('cameraStatus');
            
            // Setup audio waveform
            this.waveformCanvas = document.getElementById('audioWaveform');
            this.waveformCtx = this.waveformCanvas.getContext('2d');
            this.waveformCanvas.width = this.waveformCanvas.offsetWidth;
            this.waveformCanvas.height = this.waveformCanvas.offsetHeight;
            
            const audioStatus = document.getElementById('audioStatus');
            
            // Request media permissions
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480 },
                audio: true
            });
            
            // Setup camera
            this.cameraStream = stream.getVideoTracks()[0];
            cameraFeed.srcObject = stream;
            cameraStatus.textContent = 'Active';
            cameraStatus.style.color = '#d4af37';
            
            // Setup audio
            this.audioStream = stream.getAudioTracks()[0];
            this.setupAudioAnalysis(stream);
            audioStatus.textContent = 'Active';
            audioStatus.style.color = '#d4af37';
            
        } catch (error) {
            console.error('Media capture error:', error);
            this.updateMediaStatus('error');
        }
    }
    
    setupAudioAnalysis(stream) {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = this.audioContext.createMediaStreamSource(stream);
        this.audioAnalyser = this.audioContext.createAnalyser();
        this.audioAnalyser.fftSize = 256;
        
        source.connect(this.audioAnalyser);
        this.audioDataArray = new Uint8Array(this.audioAnalyser.frequencyBinCount);
        
        this.drawAudioWaveform();
    }
    
    drawAudioWaveform() {
        if (!this.audioAnalyser || !this.waveformCtx) return;
        
        this.audioAnalyser.getByteFrequencyData(this.audioDataArray);
        
        const canvas = this.waveformCanvas;
        const ctx = this.waveformCtx;
        const width = canvas.width;
        const height = canvas.height;
        
        // Clear canvas
        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(0, 0, width, height);
        
        // Draw waveform
        const barWidth = width / this.audioDataArray.length;
        ctx.fillStyle = '#d4af37';
        
        for (let i = 0; i < this.audioDataArray.length; i++) {
            const barHeight = (this.audioDataArray[i] / 255) * height;
            const x = i * barWidth;
            const y = height - barHeight;
            
            ctx.fillRect(x, y, barWidth - 1, barHeight);
        }
        
        this.animationFrame = requestAnimationFrame(() => this.drawAudioWaveform());
    }
    
    setupRLFeedback() {
        // Initialize RL feedback tracking
        this.selectedRating = 0;
        this.rlFeedback = {
            explicit: { comfort: 5, effectiveness: 5 },
            implicit: { interactionRate: 0, sessionDuration: 0, emotionStability: 0 },
            rl: { bufferSize: 0, learningRate: 0, rewardSignal: 0, policyConfidence: 0 }
        };
        
        // Update RL metrics periodically
        setInterval(() => {
            this.updateRLMetrics();
        }, 1000);
    }
    
    updateRLMetrics() {
        // This method is no longer needed as data comes from the server.
    }
    
    updateRLDisplay(feedback, rlAgent) {
        // Update RL metrics display
        if (feedback) {
            document.getElementById('interactionRate').textContent = (feedback.total_feedback || 0).toFixed(2);
            document.getElementById('sessionDuration').textContent = (feedback.session_duration || 0).toFixed(1) + 's';
            document.getElementById('emotionStability').textContent = (feedback.average_score || 0).toFixed(3);
        }

        if (rlAgent) {
            document.getElementById('rlBufferSize').textContent = rlAgent.buffer_size || 0;
            document.getElementById('rlLearningRate').textContent = (rlAgent.learning_rate || 0).toFixed(3);
            document.getElementById('rlRewardSignal').textContent = (rlAgent.reward_signal || 0).toFixed(3);
            document.getElementById('rlPolicyConfidence').textContent = (rlAgent.policy_confidence || 0).toFixed(3);
        }
    }
    
    updateSystemLogs(data) {
        // Update face analysis logs
        if (data.face_data) {
            const faceLog = document.getElementById('faceLogs');
            faceLog.textContent = `Face Data: V=${data.face_data.valence?.toFixed(3) || 'N/A'}, A=${data.face_data.arousal?.toFixed(3) || 'N/A'}`;
        }
        
        // Update voice analysis logs
        if (data.voice_data) {
            const voiceLog = document.getElementById('voiceLogs');
            voiceLog.textContent = `Voice Data: V=${data.voice_data.valence?.toFixed(3) || 'N/A'}, A=${data.voice_data.arousal?.toFixed(3) || 'N/A'}`;
        }
        
        // Update fusion logs
        if (data.fusion) {
            const fusionLog = document.getElementById('fusionLogs');
            fusionLog.textContent = `Fusion: V=${data.fusion.valence?.toFixed(3) || 'N/A'}, A=${data.fusion.arousal?.toFixed(3) || 'N/A'}, C=${data.fusion.confidence?.toFixed(3) || 'N/A'}`;
        }
        
        // Update music engine logs
        if (data.music_engine) {
            const musicLog = document.getElementById('musicLogs');
            musicLog.textContent = `Music Engine: ${data.music_engine.status || 'Unknown'}\nTempo: ${data.music_engine.tempo || 'N/A'} BPM\nVolume: ${data.music_engine.volume || 'N/A'}`;
        }
        
        // Update feedback logs
        if (data.feedback) {
            const feedbackLog = document.getElementById('feedbackLogs');
            feedbackLog.textContent = `Feedback: ${data.feedback.type || 'None'}\nRating: ${data.feedback.rating || 'N/A'}\nTime: ${new Date().toLocaleTimeString()}`;
        }
        
        // Update RL logs
        if (data.rl_agent) {
            const rlLog = document.getElementById('rlLogs');
            rlLog.textContent = `RL Agent: ${data.rl_agent.status || 'Unknown'}\nReward: ${data.rl_agent.reward_signal?.toFixed(3) || 'N/A'}\nAction: ${data.rl_agent.action || 'N/A'}`;
            this.updateRLDisplay(data.feedback, data.rl_agent);
        }
    }
    
    updateConnectionStatus(connected) {
        const statusOrb = document.getElementById('statusOrb');
        const statusText = document.getElementById('statusText');
        
        if (connected) {
            statusOrb.classList.add('connected');
            statusText.textContent = 'Connected';
            statusText.style.color = '#d4af37';
        } else {
            statusOrb.classList.remove('connected');
            statusText.textContent = 'Disconnected';
            statusText.style.color = '#8b0000';
        }
    }
    
    updateMediaStatus(status) {
        const cameraStatus = document.getElementById('cameraStatus');
        const audioStatus = document.getElementById('audioStatus');
        
        switch (status) {
            case 'active':
                cameraStatus.textContent = 'Active';
                audioStatus.textContent = 'Active';
                cameraStatus.style.color = '#d4af37'; // Gold for active
                audioStatus.style.color = '#d4af37';
                break;
            case 'error':
                cameraStatus.textContent = 'Error';
                audioStatus.textContent = 'Error';
                cameraStatus.style.color = '#8b0000'; // Dark red for error
                audioStatus.style.color = '#8b0000';
                break;
            default:
                cameraStatus.textContent = 'Initializing...';
                audioStatus.textContent = 'Initializing...';
                cameraStatus.style.color = '#a0a0a0'; // Grey for initializing
                audioStatus.style.color = '#a0a0a0';
        }
    }
    
    async startSession() {
        if (!this.isConnected) {
            this.showGothicNotification('Not connected to server', 'error');
            return;
        }
        
        const trajectoryType = document.getElementById('trajectorySelect').value;
        const duration = parseInt(document.getElementById('durationSlider').value);
        
        try {
            const response = await fetch('/session/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    trajectory_type: trajectoryType,
                    duration: duration
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showGothicNotification('Session started successfully', 'success');
                document.getElementById('startSessionBtn').disabled = true;
                document.getElementById('stopSessionBtn').disabled = false;
            } else {
                this.showGothicNotification(`Failed to start session: ${result.error}`, 'error');
            }
        } catch (error) {
            console.error('Start session error:', error);
            this.showGothicNotification('Failed to start session', 'error');
        }
    }
    
    async stopSession() {
        if (!this.isConnected) {
            this.showGothicNotification('Not connected to server', 'error');
            return;
        }
        
        try {
            const response = await fetch('/session/stop', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showGothicNotification('Session ended', 'info');
                document.getElementById('startSessionBtn').disabled = false;
                document.getElementById('stopSessionBtn').disabled = true;
            } else {
                this.showGothicNotification(`Failed to stop session: ${result.error}`, 'error');
            }
        } catch (error) {
            console.error('Stop session error:', error);
            this.showGothicNotification('Failed to stop session', 'error');
        }
    }
    
    handleEmotionUpdate(data) {
        console.log('Received emotion update:', data);
        
        if (data.emotion_state) {
            this.updateEmotionDisplay(data.emotion_state);
        }
        
        if (data.music_parameters) {
            this.updateMusicParameters(data.music_parameters);
        }
        
        if (data.trajectory_progress) {
            this.updateTrajectoryVisualization(data.trajectory_progress);
        }
        
        // Update system logs with the received data
        if (data.system_logs) {
            this.updateSystemLogs(data.system_logs);
        }
    }
    
    updateEmotionDisplay(emotionState) {
        const valence = emotionState.valence || 0;
        const arousal = emotionState.arousal || 0;
        const confidence = emotionState.confidence || 0;
        
        // Update emotion circle
        const emotionPoint = document.getElementById('emotionPoint');
        const circle = document.getElementById('emotionCircle');
        const circleRect = circle.getBoundingClientRect();
        const centerX = circleRect.width / 2;
        const centerY = circleRect.height / 2;
        
        // Convert valence/arousal to canvas coordinates
        const x = centerX + (valence * centerX * 0.8);
        const y = centerY - (arousal * centerY * 0.8);
        
        emotionPoint.style.left = `${x}px`;
        emotionPoint.style.top = `${y}px`;
        
        // Update values
        document.getElementById('valenceValue').textContent = valence.toFixed(3);
        document.getElementById('arousalValue').textContent = arousal.toFixed(3);
        document.getElementById('confidenceValue').textContent = confidence.toFixed(3);
    }
    
    updateTrajectoryVisualization(trajectoryProgress) {
        if (!this.trajectoryCtx) return;
        
        const canvas = this.trajectoryCanvas;
        const ctx = this.trajectoryCtx;
        const width = canvas.width;
        const height = canvas.height;
        
        // Clear canvas
        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(0, 0, width, height);
        
        // Draw grid
        this.drawTrajectoryGrid();
        
        // Draw target path (dashed line)
        if (trajectoryProgress.target_path && trajectoryProgress.target_path.length > 1) {
            ctx.strokeStyle = '#d4af37';
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.beginPath();
            
            for (let i = 0; i < trajectoryProgress.target_path.length; i++) {
                const point = trajectoryProgress.target_path[i];
                const x = (point.valence + 1) * width / 2;
                const y = (1 - point.arousal) * height / 2;
                
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            
            ctx.stroke();
            ctx.setLineDash([]);
        }
        
        // Draw actual path
        if (trajectoryProgress.actual_path && trajectoryProgress.actual_path.length > 1) {
            ctx.strokeStyle = '#8b0000';
            ctx.lineWidth = 3;
            ctx.beginPath();
            
            for (let i = 0; i < trajectoryProgress.actual_path.length; i++) {
                const point = trajectoryProgress.actual_path[i];
                const x = (point.valence + 1) * width / 2;
                const y = (1 - point.arousal) * height / 2;
                
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            
            ctx.stroke();
        }
        
        // Draw current position
        if (trajectoryProgress.actual_path && trajectoryProgress.actual_path.length > 0) {
            const currentPoint = trajectoryProgress.actual_path[trajectoryProgress.actual_path.length - 1];
            const x = (currentPoint.valence + 1) * width / 2;
            const y = (1 - currentPoint.arousal) * height / 2;
            
            ctx.fillStyle = '#d4af37';
            ctx.beginPath();
            ctx.arc(x, y, 6, 0, 2 * Math.PI);
            ctx.fill();
        }
    }
    
    drawTrajectoryGrid() {
        const canvas = this.trajectoryCanvas;
        const ctx = this.trajectoryCtx;
        const width = canvas.width;
        const height = canvas.height;
        
        ctx.strokeStyle = '#404040';
        ctx.lineWidth = 1;
        
        // Vertical lines
        for (let i = 0; i <= 4; i++) {
            const x = (i / 4) * width;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }
        
        // Horizontal lines
        for (let i = 0; i <= 4; i++) {
            const y = (i / 4) * height;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }
    }
    
    updateMusicParameters(musicParams) {
        console.log('Updating music parameters:', musicParams);
        
        // Update tempo
        if (musicParams.tempo_bpm) {
            document.getElementById('tempoValue').textContent = Math.round(musicParams.tempo_bpm);
        }
        
        // Update parameter bars
        this.updateParameterBar('rhythmComplexity', musicParams.rhythm_complexity || 0);
        this.updateParameterBar('harmonicComplexity', musicParams.chord_complexity || 0);
        this.updateParameterBar('textureDensity', musicParams.voice_density || 0);
        this.updateParameterBar('volumeLevel', musicParams.overall_volume || 0);
        
        // Update key (simplified)
        const keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
        const keyIndex = Math.floor((musicParams.brightness || 0.5) * keys.length);
        document.getElementById('keyValue').textContent = keys[keyIndex] || 'C';
    }
    
    updateParameterBar(elementId, value) {
        const element = document.getElementById(elementId);
        if (element) {
            element.style.width = `${value * 100}%`;
        }
    }
    
    updateSessionUI(active) {
        const sessionInfo = document.getElementById('sessionInfo');
        const startBtn = document.getElementById('startSessionBtn');
        const stopBtn = document.getElementById('stopSessionBtn');
        
        if (active) {
            sessionInfo.style.display = 'block';
            startBtn.disabled = true;
            stopBtn.disabled = false;
            
            // Update session info
            if (this.currentSession) {
                document.getElementById('currentGoal').textContent = this.currentSession.trajectory_type;
                document.getElementById('currentDuration').textContent = `${this.currentSession.duration} minutes`;
            }
            
            // Start progress updates
            this.updateSessionProgress();
        } else {
            sessionInfo.style.display = 'none';
            startBtn.disabled = false;
            stopBtn.disabled = true;
        }
    }
    
    updateSessionProgress() {
        if (!this.currentSession) return;
        
        const elapsed = (Date.now() - this.currentSession.startTime) / 1000;
        const totalSeconds = this.currentSession.duration * 60;
        const progress = Math.min((elapsed / totalSeconds) * 100, 100);
        
        const progressFill = document.getElementById('sessionProgress');
        const progressText = document.getElementById('progressText');
        
        progressFill.style.width = `${progress}%`;
        progressText.textContent = `${Math.round(progress)}%`;
        
        if (progress < 100) {
            setTimeout(() => this.updateSessionProgress(), 1000);
        }
    }
    
    selectRating(selectedBtn) {
        document.querySelectorAll('.rating-btn').forEach(btn => {
            btn.classList.remove('selected');
        });
        selectedBtn.classList.add('selected');
        this.selectedRating = parseInt(selectedBtn.dataset.rating);
    }
    
    async submitFeedback() {
        const rating = this.selectedRating;
        const comfort = parseInt(document.getElementById('comfortSlider').value);
        const effectiveness = parseInt(document.getElementById('effectivenessSlider').value);
        
        if (rating === 0) {
            this.showGothicNotification('Please select an emotional state rating', 'error');
            return;
        }
        
        try {
            const response = await fetch('/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    rating: rating,
                    comfort: comfort,
                    effectiveness: effectiveness,
                    emotion_state: this.emotionState // Send current emotion state
                })
            });
            
            if (response.ok) {
                this.showGothicNotification('Feedback submitted successfully', 'success');
            } else {
                const errorData = await response.json();
                this.showGothicNotification(`Failed to submit feedback: ${errorData.message}`, 'error');
            }
        } catch (error) {
            this.showGothicNotification(`Error: ${error.message}`, 'error');
        }
    }
    
    clearFeedbackForm() {
        document.querySelectorAll('.rating-btn').forEach(btn => {
            btn.classList.remove('selected');
        });
        
        document.getElementById('comfortSlider').value = 5;
        document.getElementById('effectivenessSlider').value = 5;
        document.getElementById('comfortValue').textContent = '5';
        document.getElementById('effectivenessValue').textContent = '5';
        
        this.rlFeedback.explicit.rating = 0;
    }
    
    playMusic() {
        this.socket.emit('music_control', { action: 'play' });
        this.showGothicNotification('Music playback started', 'info');
    }
    
    pauseMusic() {
        this.socket.emit('music_control', { action: 'pause' });
        this.showGothicNotification('Music paused', 'info');
    }
    
    generateMusic() {
        this.socket.emit('music_control', { action: 'regenerate' });
        this.showGothicNotification('Regenerating music parameters', 'info');
    }
    
    showGothicNotification(message, type = 'info') {
        // Create gothic notification
        const notification = document.createElement('div');
        notification.className = `gothic-notification gothic-notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas fa-${this.getNotificationIcon(type)}"></i>
                <span>${message}</span>
            </div>
        `;
        
        // Add to page
        document.body.appendChild(notification);
        
        // Animate in
        setTimeout(() => {
            notification.classList.add('show');
        }, 100);
        
        // Remove after delay
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 3000);
    }
    
    getNotificationIcon(type) {
        switch (type) {
            case 'success': return 'check-circle';
            case 'error': return 'exclamation-triangle';
            case 'warning': return 'exclamation-circle';
            default: return 'info-circle';
        }
    }
    
    updateUI() {
        // Initial UI setup
        this.updateConnectionStatus(false);
        this.updateMediaStatus('initializing');
    }
}

// Initialize the gothic client when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.emotuneClient = new EmoTuneGothicClient();
});

// Add gothic notification styles
const gothicNotificationStyles = `
<style>
.gothic-notification {
    position: fixed;
    top: 20px;
    right: 20px;
    background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%);
    border: 2px solid #404040;
    border-radius: 8px;
    padding: 15px 20px;
    color: #e0e0e0;
    font-family: 'Crimson Text', serif;
    z-index: 1000;
    transform: translateX(100%);
    transition: transform 0.3s ease;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.8);
}

.gothic-notification.show {
    transform: translateX(0);
}

.gothic-notification-success {
    border-color: #2d5a2d;
}

.gothic-notification-error {
    border-color: #8b0000;
}

.gothic-notification-warning {
    border-color: #5a4a2d;
}

.gothic-notification-info {
    border-color: #2d4a5a;
}

.notification-content {
    display: flex;
    align-items: center;
    gap: 10px;
}

.notification-content i {
    font-size: 1.2rem;
}

.gothic-notification-success .notification-content i {
    color: #4a7c4a;
}

.gothic-notification-error .notification-content i {
    color: #8b0000;
}

.gothic-notification-warning .notification-content i {
    color: #7c6a4a;
}

.gothic-notification-info .notification-content i {
    color: #4a6a7c;
}
[data-tooltip] {
    position: relative;
    cursor: help;
}
[data-tooltip]::after {
    content: attr(data-tooltip);
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    background-color: #1a1a1a;
    color: #e0e0e0;
    border: 1px solid #d4af37;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 0.9rem;
    white-space: nowrap;
    z-index: 10;
    opacity: 0;
    visibility: hidden;
    transition: opacity 0.2s, visibility 0.2s;
}
[data-tooltip]:hover::after {
    opacity: 1;
    visibility: visible;
}
</style>
`;

document.head.insertAdjacentHTML('beforeend', gothicNotificationStyles);
