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
        
        // FIX: Initialize performance optimization variables
        this._lastEmotionUpdateAt = 0;
        this._emotionUpdateIntervalMs = 100; // 10 FPS max
        this._pendingEmotionUpdate = null;
        this._emotionUpdateRafScheduled = false;
        this._trajectoryCleanupCounter = 0;
        this._performanceLogCounter = 0;
        
        // FIX: Initialize persistent trajectory storage
        this._trajectoryHistory = {
            target_path: [],
            actual_path: [],
            max_points: 1000  // Limit to prevent memory bloat
        };
        
        // NEW: Initialize time-series storage and canvas refs
        this.timeseriesCanvas = null;
        this.timeseriesCtx = null;
        this._tsHistory = [];
        this._tsMaxPoints = 1000;
        this._tsWindowSeconds = 120;
        this._lastTsDrawAt = 0;
        this._tsUpdateIntervalMs = 100; // 10 FPS max
        this._pendingTs = false;
        
        // NEW: cache last server session duration to sync progress bar
        this._serverSessionDurationSec = 0;
        // NEW: local progress update timers and drift correction
        this._progressTickerId = null;
        this._progressSyncId = null;
        this._progressDriftOffsetSec = 0;
        
        this.init();
        
        // FIX: Start periodic progress updates to prevent flickering
        this.startProgressUpdates();
    }
    
    startProgressUpdates() {
        // Clear existing timers to avoid duplicates
        if (this._progressTickerId) { clearInterval(this._progressTickerId); this._progressTickerId = null; }
        if (this._progressSyncId) { clearInterval(this._progressSyncId); this._progressSyncId = null; }
        
        // Smooth local progress at 10Hz
        this._progressTickerId = setInterval(() => {
            if (this.currentSession && (this.currentSession.server_start_time || this.currentSession.startTime)) {
                this.updateSessionProgress();
            }
        }, 100);
        
        // Periodic server sync to correct drift
        this._progressSyncId = setInterval(() => {
            this.syncSessionProgressFromServer().catch(() => {});
        }, 10000);
        
        // FIX: Add manual test function for debugging progress bar
        window.testProgressBar = () => {
            console.log('Testing progress bar...');
            const sessionInfo = document.getElementById('sessionInfo');
            const progressFill = document.getElementById('sessionProgress');
            const progressText = document.getElementById('progressText');
            
            console.log('sessionInfo:', sessionInfo);
            console.log('progressFill:', progressFill);
            console.log('progressText:', progressText);
            
            if (sessionInfo) {
                sessionInfo.style.display = 'block';
                console.log('Session info shown');
            }
            
            if (progressFill && progressText) {
                progressFill.style.width = '50%';
                progressText.textContent = '50%';
                console.log('Progress bar set to 50%');
            }
        };
        
        console.log('Progress updates started. Use window.testProgressBar() to test the progress bar manually.');
    }
    
    initializeProgressBar() {
        // Check if progress bar elements exist and are properly initialized
        const sessionInfo = document.getElementById('sessionInfo');
        const progressFill = document.getElementById('sessionProgress');
        const progressText = document.getElementById('progressText');
        
        console.log('Progress bar initialization:');
        console.log('- sessionInfo:', sessionInfo ? 'Found' : 'Missing');
        console.log('- progressFill:', progressFill ? 'Found' : 'Missing');
        console.log('- progressText:', progressText ? 'Found' : 'Missing');
        
        if (sessionInfo && progressFill && progressText) {
            console.log('✅ Progress bar elements properly initialized');
            // Set initial state
            sessionInfo.style.display = 'none';
            progressFill.style.width = '0%';
            progressText.textContent = '0%';
        } else {
            console.warn('⚠️ Some progress bar elements are missing');
        }
    }
    
    init() {
        this.setupSocketConnection();
        this.setupEventListeners();
        this.setupCanvas();
        // NEW: setup time-series canvas
        this.setupTimeSeriesCanvas();
        this.setupMediaCapture();
        this.setupRLFeedback();
        this.updateUI();
        
        // FIX: Start periodic progress updates to prevent flickering
        this.startProgressUpdates();
        
        // FIX: Initialize progress bar elements
        this.initializeProgressBar();
    }
    
    setupSocketConnection() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            console.log('Connected to EmoTune Gothic server');
            this.isConnected = true;
            this.updateConnectionStatus(true);
            // Ensure we (re)join monitoring room if a session is active
            if (this.currentSession && this.currentSession.id) {
                this.socket.emit('start_emotion_monitoring', { session_id: this.currentSession.id });
            }
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
                // Suppress monitoring toast if it fires right after a feedback toast
                const now = Date.now();
                const suppressWindowMs = 1500;
                const lastFb = this._lastFeedbackToastAt || 0;
                if (now - lastFb > suppressWindowMs) {
                    this.showGothicNotification('Media monitoring is active.', 'success');
                }
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
                // FIX: Clear trajectory history when starting new session
                if (this._trajectoryHistory) {
                    this._trajectoryHistory.target_path = [];
                    this._trajectoryHistory.actual_path = [];
                }
                
                this.currentSession = {
                    id: data.session_id,
                    trajectory_type: data.trajectory_type,
                    duration: data.duration,
                    startTime: Date.now()
                };
                this.updateSessionUI(true);
                // Immediately (re)join monitoring room with explicit session_id
                this.socket.emit('start_emotion_monitoring', { session_id: data.session_id });
            } else {
                // FIX: Clean up performance variables when session ends
                this._cleanupPerformanceVariables();
                // Preserve currentSession until backend fully stops; UI will hide the bar
                this.currentSession = null;
                this.updateSessionUI(false);
                // Keep timers alive for next session
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
            // Ensure we (re)join monitoring room if a session is active
            if (this.currentSession && this.currentSession.id) {
                this.socket.emit('start_emotion_monitoring', { session_id: this.currentSession.id });
            }
        });

        // Acknowledgements
        this.socket.on('mode_updated', (data) => {
            if (data && data.success) {
                this.showGothicNotification('Analysis mode updated', 'success');
            }
        });
        this.socket.on('thresholds_updated', (data) => {
            if (data && data.success) {
                this.showGothicNotification('Confidence thresholds updated', 'success');
            }
        });
        this.socket.on('fusion_options_updated', (data) => {
            if (data && data.success) {
                this.showGothicNotification('Fusion options updated', 'success');
            }
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
        const fusionMinConfSlider = document.getElementById('fusionMinConfSlider');
        const fusionMinConfValue = document.getElementById('fusionMinConfValue');
        const fallbackToggle = document.getElementById('fallbackToggle');

        faceConfidenceSlider.addEventListener('input', (e) => {
            faceConfidenceValue.textContent = parseFloat(e.target.value).toFixed(2);
            this.updateConfidenceThresholds();
        });

        voiceConfidenceSlider.addEventListener('input', (e) => {
            voiceConfidenceValue.textContent = parseFloat(e.target.value).toFixed(2);
            this.updateConfidenceThresholds();
        });

        fusionMinConfSlider.addEventListener('input', (e) => {
            fusionMinConfValue.textContent = parseFloat(e.target.value).toFixed(2);
            this.updateFusionOptions();
        });

        fallbackToggle.addEventListener('change', () => {
            this.updateFusionOptions();
        });

        // Ensure AudioContext resumes when tab becomes visible (fix blank waveform after pause)
        document.addEventListener('visibilitychange', async () => {
            try {
                if (document.visibilityState === 'visible' && this.audioContext && this.audioContext.state === 'suspended') {
                    await this.audioContext.resume();
                }
            } catch (e) {
                console.warn('AudioContext resume failed:', e);
            }
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

    updateFusionOptions() {
        if (!this.isConnected) return;
        const allowFallback = document.getElementById('fallbackToggle').checked;
        const fusionMinConf = parseFloat(document.getElementById('fusionMinConfSlider').value);
        this.socket.emit('update_fusion_options', {
            allow_fallback: allowFallback,
            fusion_min_conf: fusionMinConf
        });
        this.showGothicNotification(`Fusion options: fallback=${allowFallback ? 'on' : 'off'}, minC=${fusionMinConf.toFixed(2)}`, 'info');
    }

    updateAnalysisMode(mode) {
        this.socket.emit('update_analysis_mode', { mode: mode });
        console.log('Emitting analysis mode:', mode);
    }
    
    setupCanvas() {
        this.trajectoryCanvas = document.getElementById('trajectoryCanvas');
        if (!this.trajectoryCanvas) {
            console.error('Trajectory canvas element not found');
            return;
        }
        
        this.trajectoryCtx = this.trajectoryCanvas.getContext('2d');
        if (!this.trajectoryCtx) {
            console.error('Could not get canvas 2D context');
            return;
        }
        
        console.log('Canvas setup successful, size:', this.trajectoryCanvas.offsetWidth, 'x', this.trajectoryCanvas.offsetHeight);
        
        // FIXED: Set canvas size with proper initialization
        this._resizeCanvas();
        
        // FIXED: Add resize listener for responsive canvas
        window.addEventListener('resize', () => {
            this._resizeCanvas();
            this.drawTrajectoryGrid();
            // Redraw time series on resize as well
            this._resizeTimeSeriesCanvas();
            this.drawTimeSeriesAxes();
            this.updateTimeSeriesVisualization(true);
        });
        
        this.drawTrajectoryGrid();
    }
    
    _resizeCanvas() {
        // FIXED: Proper canvas sizing with bounds checking
        if (this.trajectoryCanvas) {
            const rect = this.trajectoryCanvas.getBoundingClientRect();
            this.trajectoryCanvas.width = rect.width;
            // NEW: enforce square aspect for circumplex plot
            this.trajectoryCanvas.height = rect.width;
        }
    }
    
    _validateTrajectoryProgress(trajectoryProgress) {
        // FIXED: Comprehensive validation of trajectory progress data
        if (!trajectoryProgress || typeof trajectoryProgress !== 'object') {
            return false;
        }
        
        // Validate target_path if present
        if (trajectoryProgress.target_path) {
            if (!Array.isArray(trajectoryProgress.target_path)) {
                return false;
            }
            for (const point of trajectoryProgress.target_path) {
                if (!this._validateTrajectoryPoint(point)) {
                    return false;
                }
            }
        }
        
        // Validate actual_path if present
        if (trajectoryProgress.actual_path) {
            if (!Array.isArray(trajectoryProgress.actual_path)) {
                return false;
            }
            for (const point of trajectoryProgress.actual_path) {
                if (!this._validateTrajectoryPoint(point)) {
                    return false;
                }
            }
        }
        
        // Validate current_position if present
        if (trajectoryProgress.current_position) {
            if (!this._validateTrajectoryPoint(trajectoryProgress.current_position)) {
                return false;
            }
        }
        
        return true;
    }
    
    _validateTrajectoryPoint(point) {
        // FIXED: Validate individual trajectory point
        if (!point || typeof point !== 'object') {
            return false;
        }
        
        return (
            typeof point.valence === 'number' && 
            typeof point.arousal === 'number' &&
            !isNaN(point.valence) && 
            !isNaN(point.arousal) &&
            isFinite(point.valence) && 
            isFinite(point.arousal) &&
            point.valence >= -1.0 && 
            point.valence <= 1.0 &&
            // UPDATED: arousal now symmetric in [-1,1]
            point.arousal >= -1.0 && 
            point.arousal <= 1.0
        );
    }
    
    // Normalize arousal to [-1, 1] by clamping only (server already sends [-1,1])
    _normalizeArousal(value) {
        if (typeof value !== 'number' || Number.isNaN(value) || !Number.isFinite(value)) {
            return 0;
        }
        if (value > 1) return 1;
        if (value < -1) return -1;
        return value;
    }
    
    async setupMediaCapture() {
        try {
            // Setup camera feed (backend MJPEG)
            const cameraFeed = document.getElementById('cameraFeed');
            const cameraStatus = document.getElementById('cameraStatus');
            if (cameraFeed) {
                cameraFeed.addEventListener('load', () => {
                    cameraStatus.textContent = this.emotionMonitoring ? 'Active' : 'Idle';
                    cameraStatus.style.color = this.emotionMonitoring ? '#d4af37' : '#a0a0a0';
                });
                cameraFeed.addEventListener('error', () => {
                    cameraStatus.textContent = 'Error';
                    cameraStatus.style.color = '#8b0000';
                });
            }
            
            // Teardown any previous audio resources before re-init
            if (this.animationFrame) {
                cancelAnimationFrame(this.animationFrame);
                this.animationFrame = null;
            }
            if (this.audioAnalyser) {
                this.audioAnalyser.disconnect?.();
                this.audioAnalyser = null;
            }
            if (this.audioContext) {
                try {
                    // Prefer closing to free device handles; fall back to resume if needed
                    await this.audioContext.close();
                } catch (_) {
                    try { await this.audioContext.resume(); } catch (_) {}
                }
                this.audioContext = null;
            }
            if (this.audioStream) {
                try { this.audioStream.stop?.(); } catch (_) {}
                try { this.audioStream.enabled = false; } catch (_) {}
                this.audioStream = null;
            }
            
            // Setup audio waveform
            this.waveformCanvas = document.getElementById('audioWaveform');
            this.waveformCtx = this.waveformCanvas.getContext('2d');
            this.waveformCanvas.width = this.waveformCanvas.offsetWidth;
            this.waveformCanvas.height = this.waveformCanvas.offsetHeight;
            const audioStatus = document.getElementById('audioStatus');
            
            // Request audio only to avoid camera contention
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: true
            });
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
            const total = feedback.total_feedback || 0;
            const durSec = feedback.session_duration || 0;
            const perMinServer = feedback.interaction_rate;
            const perMin = (typeof perMinServer === 'number') ? perMinServer : (durSec > 0 ? (total / (durSec / 60.0)) : 0);
            document.getElementById('interactionRate').textContent = perMin.toFixed(2);
            document.getElementById('sessionDuration').textContent = durSec.toFixed(1) + 's';
            // Prefer server stability metric if present in last emotion_update payload cache
            const stability = (this._lastEmotionMetrics && typeof this._lastEmotionMetrics.stability === 'number')
                ? this._lastEmotionMetrics.stability
                : (feedback.average_score || 0);
            document.getElementById('emotionStability').textContent = Number(stability).toFixed(3);
            // Keep server duration cache only for diagnostics; progress is now locally driven
            this._serverSessionDurationSec = durSec;
        }

        if (rlAgent) {
            document.getElementById('rlBufferSize').textContent = rlAgent.buffer_size || 0;
            document.getElementById('rlLearningRate').textContent = (Number(rlAgent.learning_rate) || 0).toFixed(6);
            document.getElementById('rlRewardSignal').textContent = (Number(rlAgent.reward_signal) || 0).toFixed(3);
            document.getElementById('rlPolicyConfidence').textContent = (Number(rlAgent.policy_confidence) || 0).toFixed(3);
            // RL Details panel
            const latestReward = document.getElementById('rlLatestReward');
            const rlAlpha = document.getElementById('rlAlpha');
            const tableBody = document.getElementById('rlActionTable');
            if (latestReward) latestReward.textContent = (Number(rlAgent.reward_signal) || 0).toFixed(3);
            if (rlAlpha) rlAlpha.textContent = (Number(rlAgent.policy_confidence) || 0).toFixed(3);
            if (tableBody) {
                tableBody.innerHTML = '';
                const params = rlAgent.params || {};
                const actions = rlAgent.action || {};
                Object.keys(params).forEach((key) => {
                    const tr = document.createElement('tr');
                    const tdK = document.createElement('td');
                    const tdA = document.createElement('td');
                    const tdP = document.createElement('td');
                    const label = key.replace(/_/g, ' ').replace(/\b\w/g, (m) => m.toUpperCase());
                    tdK.textContent = label;
                    const aVal = actions.hasOwnProperty(key) ? actions[key] : '-';
                    const pVal = params.hasOwnProperty(key) ? params[key] : '-';
                    tdA.textContent = (typeof aVal === 'number') ? aVal.toFixed(3) : (aVal ?? '-');
                    tdP.textContent = (typeof pVal === 'number') ? pVal.toFixed(3) : (pVal ?? '-');
                    tr.appendChild(tdK); tr.appendChild(tdA); tr.appendChild(tdP);
                    tableBody.appendChild(tr);
                });
            }
        }
    }
    
    updateSystemLogs(data) {
        // FIX: Batch all DOM updates to reduce reflows and improve performance
        requestAnimationFrame(() => {
            // Update face analysis logs
            if (data.face_data) {
                const faceLog = document.getElementById('faceLogs');
                if (faceLog) {
                    faceLog.textContent = `Face Data: V=${data.face_data.valence?.toFixed(3) || 'N/A'}, A=${data.face_data.arousal?.toFixed(3) || 'N/A'}`;
                }
            }
            
            // Update voice analysis logs
            if (data.voice_data) {
                const voiceLog = document.getElementById('voiceLogs');
                if (voiceLog) {
                    voiceLog.textContent = `Voice Data: V=${data.voice_data.valence?.toFixed(3) || 'N/A'}, A=${data.voice_data.arousal?.toFixed(3) || 'N/A'}`;
                }
            }
            
            // Update fusion logs
            if (data.fusion) {
                const fusionLog = document.getElementById('fusionLogs');
                if (fusionLog) {
                    fusionLog.textContent = `Fusion: V=${data.fusion.valence?.toFixed(3) || 'N/A'}, A=${data.fusion.arousal?.toFixed(3) || 'N/A'}, C=${data.fusion.confidence?.toFixed(3) || 'N/A'}`;
                }
            }
            
            // Update music engine logs
            if (data.music_engine) {
                const musicLog = document.getElementById('musicLogs');
                if (musicLog) {
                    musicLog.textContent = `Music Engine: ${data.music_engine.status || 'Unknown'}\nTempo: ${data.music_engine.tempo || 'N/A'} BPM\nVolume: ${data.music_engine.volume || 'N/A'}`;
                }
            }
            
            // Update feedback logs
            if (data.feedback) {
                const feedbackLog = document.getElementById('feedbackLogs');
                if (feedbackLog) {
                    feedbackLog.textContent = `Feedback: ${data.feedback.type || 'None'}\nRating: ${data.feedback.rating || 'N/A'}\nTime: ${new Date().toLocaleTimeString()}`;
                }
            }
            
            // Update RL logs
            if (data.rl_agent) {
                const rlLog = document.getElementById('rlLogs');
                if (rlLog) {
                    const act = data.rl_agent.action || {};
                    const actionStr = Object.keys(act).length
                        ? Object.entries(act).map(([k, v]) => `${k}:${(typeof v === 'number' ? v.toFixed(3) : v)}`).join(', ')
                        : 'N/A';
                    rlLog.textContent = `RL Agent: ${data.rl_agent.status || 'Unknown'}\nReward: ${(Number(data.rl_agent.reward_signal) || 0).toFixed(3)}\nAction: ${actionStr}`;
                }
                this.updateRLDisplay(data.feedback, data.rl_agent);
            }
        });
    }
    
    updateCaptureIndicators(systemLogs, emotionState, fullPayload) {
        // FIX: Batch DOM updates to reduce reflows
        requestAnimationFrame(() => {
            // Pull raw modality data
            const face = systemLogs.face_data || {};
            const voice = systemLogs.voice_data || {};
            // Try to infer used sources from payload (if available in fused result). Fallback to confidence>0 threshold
            let faceUsed = '-';
            let voiceUsed = '-';
            if (systemLogs.fusion_sources) {
                faceUsed = systemLogs.fusion_sources.face ? 'Yes' : 'No';
                voiceUsed = systemLogs.fusion_sources.voice ? 'Yes' : 'No';
            } else if (fullPayload && fullPayload.fusion && fullPayload.fusion.sources) {
                faceUsed = fullPayload.fusion.sources.face ? 'Yes' : 'No';
                voiceUsed = fullPayload.fusion.sources.voice ? 'Yes' : 'No';
            } else {
                faceUsed = (typeof face.confidence === 'number' && face.confidence > 0.05) ? 'Yes' : 'No';
                voiceUsed = (typeof voice.confidence === 'number' && voice.confidence > 0.05) ? 'Yes' : 'No';
            }

            // Update DOM with null checks
            const set = (id, val) => { 
                const el = document.getElementById(id); 
                if (el) el.textContent = val; 
            };
            set('faceV', this._fmt(face.valence));
            set('faceA', this._fmt(face.arousal));
            set('faceC', this._fmt(face.confidence));
            set('voiceV', this._fmt(voice.valence));
            set('voiceA', this._fmt(voice.arousal));
            set('voiceC', this._fmt(voice.confidence));
            set('faceUsed', faceUsed);
            set('voiceUsed', voiceUsed);
        });
    }

    _fmt(val) {
        if (val === null || val === undefined || Number.isNaN(val)) return '-';
        const num = Number(val);
        return Number.isFinite(num) ? num.toFixed(3) : '-';
    }
    
    _cleanupPerformanceVariables() {
        // FIX: Clean up performance optimization variables to prevent memory leaks
        this._pendingEmotionUpdate = null;
        this._emotionUpdateRafScheduled = false;
        this._trajectoryCleanupCounter = 0;
        this._performanceLogCounter = 0;
        this._pendingTrajectory = null;
        this._trajRafScheduled = false;
        
        // FIX: Clear trajectory history when session ends
        if (this._trajectoryHistory) {
            this._trajectoryHistory.target_path = [];
            this._trajectoryHistory.actual_path = [];
            // FIX: Use server session start time if available, otherwise use current time
            this._trajectoryHistory.session_start_time = (this.currentSession && this.currentSession.server_start_time) 
                ? this.currentSession.server_start_time * 1000 
                : Date.now();
        }
        
        // Do not clear local progress timers here; they are reused across sessions
        // Force garbage collection if available
        if (window.gc) window.gc();
    }
    
    _logPerformanceMetrics() {
        // FIX: Add performance monitoring to help debug future issues
        if (this._performanceLogCounter === undefined) this._performanceLogCounter = 0;
        this._performanceLogCounter++;
        
        // Log performance metrics every 100 updates
        if (this._performanceLogCounter >= 100) {
            this._performanceLogCounter = 0;
            
            const memoryInfo = performance.memory;
            if (memoryInfo) {
                console.log(`Performance Metrics - Used: ${Math.round(memoryInfo.usedJSHeapSize / 1024 / 1024)}MB, Total: ${Math.round(memoryInfo.totalJSHeapSize / 1024 / 1024)}MB`);
            }
            
            // Log throttling statistics
            const now = performance.now();
            const timeSinceLastUpdate = now - this._lastEmotionUpdateAt;
            console.log(`Throttling Stats - Time since last update: ${timeSinceLastUpdate.toFixed(2)}ms, Target interval: ${this._emotionUpdateIntervalMs}ms`);
        }
    }
    
    _updateTrajectoryHistory(trajectoryProgress) {
        // FIXED: Maintain persistent trajectory history for continuous visualization
        if (!this._trajectoryHistory) {
            this._trajectoryHistory = {
                target_path: [],
                actual_path: [],
                max_points: 300,   // Realistic limit for typical sessions
                fade_start_threshold: 150,  // Start fading after ~5 minutes
                fade_duration: 150,  // Fade over the same duration
                // FIX: Use server session start time if available, otherwise use current time
                session_start_time: (this.currentSession && this.currentSession.server_start_time) 
                    ? this.currentSession.server_start_time * 1000 
                    : Date.now()
            };
        }
        
        // FIXED: Validate trajectory progress data before processing
        if (!this._validateTrajectoryProgress(trajectoryProgress)) {
            console.warn('Invalid trajectory progress data received');
            return;
        }
        
        // Add new target path points (if provided)
        if (trajectoryProgress.target_path && Array.isArray(trajectoryProgress.target_path)) {
            // Replace with latest snapshot to avoid duplicate/fan connections
            this._trajectoryHistory.target_path = trajectoryProgress.target_path.slice(0, this._trajectoryHistory.max_points);
            this._addFadeEffectToOldPoints('target_path');
        }
        
        // Add new actual path points (if provided)
        if (trajectoryProgress.actual_path && Array.isArray(trajectoryProgress.actual_path)) {
            // Replace with latest snapshot to maintain correct sequential connections
            this._trajectoryHistory.actual_path = trajectoryProgress.actual_path.slice(0, this._trajectoryHistory.max_points);
            this._addFadeEffectToOldPoints('actual_path');
        }
        
        // If no path data provided, try to add current position as a single point
        if (trajectoryProgress.current_position && 
            typeof trajectoryProgress.current_position.valence === 'number' && 
            typeof trajectoryProgress.current_position.arousal === 'number') {
            
            const currentPoint = {
                valence: trajectoryProgress.current_position.valence,
                arousal: trajectoryProgress.current_position.arousal,
                timestamp: Date.now()
            };
            
            this._trajectoryHistory.actual_path.push(currentPoint);
            // FIX: Use smarter compression instead of simple truncation
            this._compressPathIfNeeded('actual_path');
        }
    }
    
    _compressPathIfNeeded(pathType) {
        // FIX: Gradual aging-out system instead of sudden compression
        const path = this._trajectoryHistory[pathType];
        if (path.length <= this._trajectoryHistory.max_points) {
            return; // No aging needed yet
        }
        
        // Start gradual aging when we exceed the fade threshold
        if (path.length > this._trajectoryHistory.fade_start_threshold) {
            this._applyGradualAging(pathType);
        }
    }
    
    _applyGradualAging(pathType) {
        // FIXED: Gradually fade out old trajectory points with improved removal rate
        const path = this._trajectoryHistory[pathType];
        const fadeDuration = this._trajectoryHistory.fade_duration;
        
        // FIXED: Calculate how many points to gradually remove - more aggressive cleanup
        const excessPoints = path.length - this._trajectoryHistory.max_points;
        const pointsToRemove = Math.min(excessPoints, Math.max(1, Math.floor(excessPoints * 0.2))); // Remove 20% of excess points
        
        if (pointsToRemove > 0) {
            // Remove oldest points gradually (from the beginning)
            this._trajectoryHistory[pathType] = path.slice(pointsToRemove);
            
            // Add fade-out effect to the remaining oldest points
            this._addFadeEffectToOldPoints(pathType);
        }
    }
    
    _addFadeEffectToOldPoints(pathType) {
        // FIXED: Add visual fade effect to old trajectory points with correct logic
        const path = this._trajectoryHistory[pathType];
        const fadeDuration = this._trajectoryHistory.fade_duration;
        
        // FIXED: Add alpha/opacity property to points for fade effect - corrected calculation
        for (let i = 0; i < Math.min(path.length, fadeDuration); i++) {
            const fadeRatio = (fadeDuration - i) / fadeDuration; // FIXED: Inverted ratio for correct fade (oldest = 1, newest = 0)
            const alpha = 0.3 + (fadeRatio * 0.7); // Fade from 0.3 to 1.0 opacity (oldest to newest)
            path[i].alpha = alpha;
        }
        
        // Ensure newer points are fully opaque
        for (let i = fadeDuration; i < path.length; i++) {
            path[i].alpha = 1.0;
        }
    }
    
    _douglasPeuckerCompression(points, tolerance) {
        // FIX: Implement Douglas-Peucker algorithm for path compression
        if (points.length <= 2) return points;
        
        // Find the point with maximum distance from line segment
        let maxDistance = 0;
        let maxIndex = 0;
        const start = points[0];
        const end = points[points.length - 1];
        
        for (let i = 1; i < points.length - 1; i++) {
            const distance = this._pointToLineDistance(points[i], start, end);
            if (distance > maxDistance) {
                maxDistance = distance;
                maxIndex = i;
            }
        }
        
        // If max distance is greater than tolerance, recursively compress
        if (maxDistance > tolerance) {
            const firstHalf = this._douglasPeuckerCompression(points.slice(0, maxIndex + 1), tolerance);
            const secondHalf = this._douglasPeuckerCompression(points.slice(maxIndex), tolerance);
            // Combine results, avoiding duplicate point at maxIndex
            return firstHalf.slice(0, -1).concat(secondHalf);
        } else {
            // Return just the start and end points
            return [start, end];
        }
    }
    
    _pointToLineDistance(point, lineStart, lineEnd) {
        // FIX: Calculate distance from point to line segment
        const A = point.valence - lineStart.valence;
        const B = point.arousal - lineStart.arousal;
        const C = lineEnd.valence - lineStart.valence;
        const D = lineEnd.arousal - lineStart.arousal;
        
        const dot = A * C + B * D;
        const lenSq = C * C + D * D;
        
        if (lenSq === 0) {
            // Line segment is actually a point
            return Math.sqrt(A * A + B * B);
        }
        
        let param = dot / lenSq;
        
        let xx, yy;
        if (param < 0) {
            xx = lineStart.valence;
            yy = lineStart.arousal;
        } else if (param > 1) {
            xx = lineEnd.valence;
            yy = lineEnd.arousal;
        } else {
            xx = lineStart.valence + param * C;
            yy = lineStart.arousal + param * D;
        }
        
        const dx = point.valence - xx;
        const dy = point.arousal - yy;
        return Math.sqrt(dx * dx + dy * dy);
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
        
        // Reset visuals for a fresh session
        this.resetVisualizations();
        
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
                
                // FIX: Store session info with server-provided start time for proper synchronization
                this.currentSession = { 
                    id: result.session_id, 
                    trajectory_type: trajectoryType, 
                    duration: duration,
                    // FIX: Use server-provided start time to avoid timer conflicts
                    server_start_time: result.session_start_time || Date.now() / 1000
                };
                // Reset drift and kick ticker immediately for a fresh 0% start
                this._progressDriftOffsetSec = 0;
                this.startProgressUpdates();
                
                // FIX: Update UI to show session info and progress bar
                this.updateSessionUI(true);
                
                if (this.socket) {
                    this.socket.emit('start_emotion_monitoring', { session_id: result.session_id });
                }
                // Ensure microphone visualizer is active after session start
                try { await this.setupMediaCapture(); } catch (_) {}
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
                // FIX: Clean up performance variables when manually stopping session
                this._cleanupPerformanceVariables();
                this.showGothicNotification('Session ended', 'info');
                
                // FIX: Update UI to hide session info and progress bar
                this.updateSessionUI(false);
                
            } else {
                this.showGothicNotification(`Failed to stop session: ${result.error}`, 'error');
            }
        } catch (error) {
            console.error('Stop session error:', error);
            this.showGothicNotification('Failed to stop session', 'error');
        }
    }
    
    handleEmotionUpdate(data) {
		// FIX: Implement global throttling to prevent UI freezing
		const now = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
		
		// Initialize throttling variables if not set
		if (this._lastEmotionUpdateAt === undefined) this._lastEmotionUpdateAt = 0;
		if (this._emotionUpdateIntervalMs === undefined) this._emotionUpdateIntervalMs = 100; // 10 FPS max
		
		// Throttle all emotion updates to prevent UI freezing
		if (now - this._lastEmotionUpdateAt < this._emotionUpdateIntervalMs) {
			// Save latest payload and schedule a batched update
			this._pendingEmotionUpdate = data;
			if (!this._emotionUpdateRafScheduled) {
				this._emotionUpdateRafScheduled = true;
				requestAnimationFrame(() => {
					this._emotionUpdateRafScheduled = false;
					const pending = this._pendingEmotionUpdate;
					this._pendingEmotionUpdate = null;
					if (pending) this.handleEmotionUpdate(pending);
				});
			}
			return;
		}
		this._lastEmotionUpdateAt = now;
		
		// FIX: Log performance metrics
		this._logPerformanceMetrics();
		
		// Process latest emotion update (logging removed for performance)
		
		if (data.emotion_state) {
			this.updateEmotionDisplay(data.emotion_state);
			// cache for feedback payloads
			this.emotionState = data.emotion_state;
			// NEW: update time-series buffer
			this._updateTimeSeries(data.emotion_state);
		}
		
		// NEW: cache emotion_metrics
		if (data.emotion_metrics) {
			this._lastEmotionMetrics = data.emotion_metrics;
		}
		
		if (data.music_parameters) {
			this.updateMusicParameters(data.music_parameters);
		}
		
		if (data.trajectory_progress) {
			this.updateTrajectoryVisualization(data.trajectory_progress);
			// Update target/deviation text fields
			const tp = data.trajectory_progress;
			if (tp.current_target && typeof tp.current_target.valence === 'number' && typeof tp.current_target.arousal === 'number') {
				const tgt = `(${tp.current_target.valence.toFixed(2)}, ${tp.current_target.arousal.toFixed(2)})`;
				document.getElementById('targetEmotion').textContent = tgt;
			}
			if (typeof tp.deviation === 'number') {
				document.getElementById('trajectoryDeviation').textContent = tp.deviation.toFixed(3);
			}
		}
        
        // Update system logs with the received data
        if (data.system_logs) {
            this.updateSystemLogs(data.system_logs);
            this.updateCaptureIndicators(data.system_logs, data.emotion_state, data);
        }
    }
    
    updateEmotionDisplay(emotionState) {
        const valence = emotionState.valence || 0;
        const arousal = emotionState.arousal || 0;
        const confidence = emotionState.confidence || 0;
        
        // FIX: Batch DOM operations to reduce reflows
        requestAnimationFrame(() => {
            // Update emotion circle
            const emotionPoint = document.getElementById('emotionPoint');
            const circle = document.getElementById('emotionCircle');
            const circleRect = circle.getBoundingClientRect();
            const centerX = circleRect.width / 2;
            const centerY = circleRect.height / 2;
            
            // Convert valence/arousal to canvas coordinates (both already in [-1,1])
            const x = centerX + (valence * centerX * 0.8);
            const y = centerY - (this._normalizeArousal(arousal) * centerY * 0.8);
            
            emotionPoint.style.left = `${x}px`;
            emotionPoint.style.top = `${y}px`;
            
            // Update values
            document.getElementById('valenceValue').textContent = valence.toFixed(3);
            document.getElementById('arousalValue').textContent = arousal.toFixed(3);
            document.getElementById('confidenceValue').textContent = confidence.toFixed(3);
        });
    }
    
    updateTrajectoryVisualization(trajectoryProgress) {
		if (!this.trajectoryCtx) {
			console.warn('Trajectory context not available');
			return;
		}
		
		console.log('Updating trajectory visualization with data:', trajectoryProgress);
		
		// FIXED: Enhanced throttling with adaptive intervals based on path complexity
		const now = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
		if (this._lastTrajDrawAt === undefined) this._lastTrajDrawAt = 0;
		
		// FIXED: Adaptive interval based on path complexity for better performance
		const pathComplexity = (this._trajectoryHistory?.actual_path?.length || 0) + 
		                      (this._trajectoryHistory?.target_path?.length || 0);
		const adaptiveInterval = Math.max(50, Math.min(200, 100 + pathComplexity * 0.5));
		
		if (now - this._lastTrajDrawAt < adaptiveInterval) {
			// Save latest payload and schedule a redraw
			this._pendingTrajectory = trajectoryProgress;
			if (!this._trajRafScheduled) {
				this._trajRafScheduled = true;
				requestAnimationFrame(() => {
					this._trajRafScheduled = false;
					const pending = this._pendingTrajectory;
					this._pendingTrajectory = null;
					if (pending) this.updateTrajectoryVisualization(pending);
				});
			}
			return;
		}
		this._lastTrajDrawAt = now;
		
		// FIX: Update persistent trajectory history
		this._updateTrajectoryHistory(trajectoryProgress);
		
		// FIX: Add memory cleanup for trajectory paths to prevent accumulation
		if (this._trajectoryCleanupCounter === undefined) this._trajectoryCleanupCounter = 0;
		this._trajectoryCleanupCounter++;
		
		// Clean up old path data every 100 updates to prevent memory bloat
		if (this._trajectoryCleanupCounter >= 100) {
			this._trajectoryCleanupCounter = 0;
			// Force garbage collection hint for path data
			if (window.gc) window.gc();
		}
		
		const canvas = this.trajectoryCanvas;
		const ctx = this.trajectoryCtx;
		const width = canvas.width;
		const height = canvas.height;
		
		// FIXED: Utility to decimate long paths to a reasonable number of points with improved algorithm
		const decimate = (points, maxPoints) => {
			if (!Array.isArray(points)) return [];
			const n = points.length;
			if (n <= maxPoints) return points;
			
			// FIXED: Improved decimation algorithm to ensure exact number of points
			const step = n / maxPoints;
			const out = [];
			
			for (let i = 0; i < maxPoints; i++) {
				const index = Math.floor(i * step);
				out.push(points[index]);
			}
			
			// Ensure last point is included
			if (out.length > 0 && out[out.length - 1] !== points[n - 1]) {
				out[out.length - 1] = points[n - 1];
			}
			
			return out;
		};
		
		// FIX: Use persistent trajectory history instead of just current data
		const targetPath = decimate(this._trajectoryHistory.target_path || [], 300);
		const actualPath = decimate(this._trajectoryHistory.actual_path || [], 300);
		
		console.log(`Drawing paths: target=${targetPath.length} points, actual=${actualPath.length} points`);
		
		// Clear canvas
		ctx.fillStyle = '#1a1a1a';
		ctx.fillRect(0, 0, width, height);
		
		// Draw grid
		this.drawTrajectoryGrid();
		
		// FIXED: Draw target path (dashed line) with fade effect and full canvas coordinates
		if (targetPath.length > 1) {
			ctx.lineWidth = 2;
			ctx.setLineDash([5, 5]);
			
			// Draw with fade effect
			for (let i = 0; i < targetPath.length - 1; i++) {
				const point1 = targetPath[i];
				const point2 = targetPath[i + 1];
				const alpha1 = point1.alpha !== undefined ? point1.alpha : 1.0;
				const alpha2 = point2.alpha !== undefined ? point2.alpha : 1.0;
				const a1 = this._normalizeArousal(point1.arousal);
				const a2 = this._normalizeArousal(point2.arousal);
				
				// FIXED: Create gradient for this line segment with full canvas coordinate transformation
				const gradient = ctx.createLinearGradient(
					(point1.valence + 1) * width / 2, (1 - (a1 + 1) / 2) * height,
					(point2.valence + 1) * width / 2, (1 - (a2 + 1) / 2) * height
				);
				
				const color1 = `rgba(212, 175, 55, ${alpha1})`; // #d4af37 with alpha
				const color2 = `rgba(212, 175, 55, ${alpha2})`;
				
				gradient.addColorStop(0, color1);
				gradient.addColorStop(1, color2);
				
				ctx.strokeStyle = gradient;
				ctx.beginPath();
				ctx.moveTo((point1.valence + 1) * width / 2, (1 - (a1 + 1) / 2) * height);
				ctx.lineTo((point2.valence + 1) * width / 2, (1 - (a2 + 1) / 2) * height);
				ctx.stroke();
			}
			ctx.setLineDash([]);
		}
		
		// FIXED: Draw actual path with fade effect and full canvas coordinates
		if (actualPath.length > 1) {
			ctx.lineWidth = 3;
			
			// Draw with fade effect
			for (let i = 0; i < actualPath.length - 1; i++) {
				const point1 = actualPath[i];
				const point2 = actualPath[i + 1];
				const alpha1 = point1.alpha !== undefined ? point1.alpha : 1.0;
				const alpha2 = point2.alpha !== undefined ? point2.alpha : 1.0;
				const a1 = this._normalizeArousal(point1.arousal);
				const a2 = this._normalizeArousal(point2.arousal);
				
				// FIXED: Create gradient for this line segment with full canvas coordinate transformation
				const gradient = ctx.createLinearGradient(
					(point1.valence + 1) * width / 2, (1 - (a1 + 1) / 2) * height,
					(point2.valence + 1) * width / 2, (1 - (a2 + 1) / 2) * height
				);
				
				const color1 = `rgba(139, 0, 0, ${alpha1})`; // #8b0000 with alpha
				const color2 = `rgba(139, 0, 0, ${alpha2})`;
				
				gradient.addColorStop(0, color1);
				gradient.addColorStop(1, color2);
				
				ctx.strokeStyle = gradient;
				ctx.beginPath();
				ctx.moveTo((point1.valence + 1) * width / 2, (1 - (a1 + 1) / 2) * height);
				ctx.lineTo((point2.valence + 1) * width / 2, (1 - (a2 + 1) / 2) * height);
				ctx.stroke();
			}
		}
		
		// FIXED: Draw current position with full canvas coordinates
		if (actualPath.length > 0) {
			const currentPoint = actualPath[actualPath.length - 1];
			const aN = this._normalizeArousal(currentPoint.arousal);
			const x = (currentPoint.valence + 1) * width / 2;
			const y = (1 - (aN + 1) / 2) * height; // UPDATED: normalized arousal mapping
			ctx.fillStyle = '#d4af37';
			ctx.beginPath();
			ctx.arc(x, y, 6, 0, 2 * Math.PI);
			ctx.fill();
		}
	}
    
    drawTrajectoryGrid() {
        const canvas = this.trajectoryCanvas;
        const ctx = this.trajectoryCtx;
        
        if (!canvas || !ctx) {
            console.error('Canvas or context not available for grid drawing');
            return;
        }
        
        const width = canvas.width;
        const height = canvas.height;
        
        console.log('Drawing grid on canvas:', width, 'x', height);
        
        ctx.strokeStyle = '#404040';
        ctx.lineWidth = 1;
        
        // FIXED: Vertical lines (valence axis) - correctly aligned
        for (let i = 0; i <= 4; i++) {
            const x = (i / 4) * width;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }
        
        // UPDATED: Horizontal lines (arousal axis) for symmetric range [-1,1]
        const yTicks = [-1, -0.5, 0, 0.5, 1];
        for (let i = 0; i < yTicks.length; i++) {
            const y = (1 - (yTicks[i] + 1) / 2) * height;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }
        
        // UPDATED: Axis labels and tick values (symmetric)
        ctx.fillStyle = '#a0a0a0';
        ctx.font = '12px "Crimson Text", serif';
        ctx.textAlign = 'center';
        
        // Valence ticks: -1.0 .. 1.0
        const vTicks = [-1.0, -0.5, 0.0, 0.5, 1.0];
        for (let i = 0; i < vTicks.length; i++) {
            const x = ((vTicks[i] + 1) / 2) * width;
            ctx.fillText(vTicks[i].toFixed(1), x, height - 6);
        }
        
        // Arousal ticks: -1.0 .. 1.0
        ctx.textAlign = 'right';
        for (let i = 0; i < yTicks.length; i++) {
            const y = (1 - (yTicks[i] + 1) / 2) * height;
            ctx.fillText(yTicks[i].toFixed(1), 34, y + 4);
        }
        
        // Axis titles
        ctx.textAlign = 'center';
        ctx.fillText('Valence', width / 2, height - 20);
        ctx.save();
        ctx.translate(14, height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Arousal', 0, 0);
        ctx.restore();
    }
    
    // DEBUG: Function to test trajectory drawing manually
    testTrajectoryDrawing() {
        console.log('Testing trajectory drawing...');
        
        if (!this.trajectoryCtx) {
            console.error('No trajectory context available');
            return;
        }
        
        // Clear canvas
        const canvas = this.trajectoryCanvas;
        const ctx = this.trajectoryCtx;
        const width = canvas.width;
        const height = canvas.height;
        
        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(0, 0, width, height);
        
        // Draw grid
        this.drawTrajectoryGrid();
        
        // Create test trajectory data
        const testActualPath = [
            { valence: -0.8, arousal: 0.2 },
            { valence: -0.4, arousal: 0.4 },
            { valence: 0.0, arousal: 0.6 },
            { valence: 0.4, arousal: 0.5 },
            { valence: 0.8, arousal: 0.3 }
        ];
        
        const testTargetPath = [
            { valence: -0.7, arousal: 0.3 },
            { valence: -0.3, arousal: 0.5 },
            { valence: 0.1, arousal: 0.7 },
            { valence: 0.5, arousal: 0.6 },
            { valence: 0.9, arousal: 0.4 }
        ];
        
        // Draw target path (dashed)
        if (testTargetPath.length > 1) {
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.strokeStyle = '#d4af37';
            
            ctx.beginPath();
            for (let i = 0; i < testTargetPath.length - 1; i++) {
                const point1 = testTargetPath[i];
                const point2 = testTargetPath[i + 1];
                
                const x1 = (point1.valence + 1) * width / 2;
                const y1 = (1 - point1.arousal) * height;
                const x2 = (point2.valence + 1) * width / 2;
                const y2 = (1 - point2.arousal) * height;
                
                if (i === 0) ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);
            }
            ctx.stroke();
            ctx.setLineDash([]);
        }
        
        // Draw actual path (solid)
        if (testActualPath.length > 1) {
            ctx.lineWidth = 3;
            ctx.strokeStyle = '#8b0000';
            
            ctx.beginPath();
            for (let i = 0; i < testActualPath.length - 1; i++) {
                const point1 = testActualPath[i];
                const point2 = testActualPath[i + 1];
                
                const x1 = (point1.valence + 1) * width / 2;
                const y1 = (1 - point1.arousal) * height;
                const x2 = (point2.valence + 1) * width / 2;
                const y2 = (1 - point2.arousal) * height;
                
                if (i === 0) ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);
            }
            ctx.stroke();
        }
        
        // Draw current position
        const currentPoint = testActualPath[testActualPath.length - 1];
        const x = (currentPoint.valence + 1) * width / 2;
        const y = (1 - currentPoint.arousal) * height;
        
        ctx.fillStyle = '#d4af37';
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, 2 * Math.PI);
        ctx.fill();
        
        console.log('Test trajectory drawing completed');
    }
    
    updateMusicParameters(musicParams) {
		// Update music parameter UI (logging removed for performance)
		
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
        console.log(`updateSessionUI called with active: ${active}`);
        
        const sessionInfo = document.getElementById('sessionInfo');
        const startBtn = document.getElementById('startSessionBtn');
        const stopBtn = document.getElementById('stopSessionBtn');
        
        if (!sessionInfo) {
            console.warn('sessionInfo element not found');
            return;
        }
        
        if (active) {
            console.log('Showing session info');
            sessionInfo.style.display = 'block';
            startBtn.disabled = true;
            stopBtn.disabled = false;
            
            // Update session info
            if (this.currentSession) {
                const currentGoal = document.getElementById('currentGoal');
                const currentDuration = document.getElementById('currentDuration');
                
                if (currentGoal) currentGoal.textContent = this.currentSession.trajectory_type;
                if (currentDuration) currentDuration.textContent = `${this.currentSession.duration} minutes`;
                
                console.log(`Session info updated: ${this.currentSession.trajectory_type}, ${this.currentSession.duration} minutes`);
            }
            
            // Reset progress bar to 0 and restart ticker for a fresh session
            const progressFill = document.getElementById('sessionProgress');
            const progressText = document.getElementById('progressText');
            if (progressFill) progressFill.style.width = '0%';
            if (progressText) progressText.textContent = '0%';
            this._progressDriftOffsetSec = 0;
            this.startProgressUpdates();
        } else {
            console.log('Hiding session info');
            sessionInfo.style.display = 'none';
            startBtn.disabled = false;
            stopBtn.disabled = true;
            // Reset UI on stop
            this.resetVisualizations();
        }
    }
    
    // FIX: Update session progress using local timer synced to server start time
    async updateSessionProgress() {
        if (!this.currentSession) return;
        const progressFill = document.getElementById('sessionProgress');
        const progressText = document.getElementById('progressText');
        if (!progressFill || !progressText) return;
        const totalSeconds = Math.max(0, (this.currentSession.duration || 0) * 60);
        if (totalSeconds <= 0) return;
        const nowSec = Date.now() / 1000;
        const baseStart = this.currentSession.server_start_time || (this.currentSession.startTime ? this.currentSession.startTime / 1000 : nowSec);
        const elapsed = Math.max(0, (nowSec - baseStart) + (this._progressDriftOffsetSec || 0));
        const pct = Math.round(Math.min(Math.max(elapsed / totalSeconds, 0), 1) * 100);
        progressFill.style.width = `${pct}%`;
        progressText.textContent = `${pct}%`;
    }
    
    selectRating(selectedBtn) {
        document.querySelectorAll('.rating-btn').forEach(btn => {
            btn.classList.remove('selected');
        });
        selectedBtn.classList.add('selected');
        this.selectedRating = parseInt(selectedBtn.dataset.rating);
		// Enable sliders now that a base sentiment is chosen and reset to neutral
		const comfortSlider = document.getElementById('comfortSlider');
		const effectivenessSlider = document.getElementById('effectivenessSlider');
		const comfortValue = document.getElementById('comfortValue');
		const effectivenessValue = document.getElementById('effectivenessValue');
		if (comfortSlider && effectivenessSlider) {
			comfortSlider.disabled = false;
			effectivenessSlider.disabled = false;
			comfortSlider.value = 5;
			effectivenessSlider.value = 5;
			if (comfortValue) comfortValue.textContent = '5';
			if (effectivenessValue) effectivenessValue.textContent = '5';
		}
    }
    
    async submitFeedback() {
        const rating = this.selectedRating;
        const comfortSlider = document.getElementById('comfortSlider');
        const effectivenessSlider = document.getElementById('effectivenessSlider');
        const comfort = parseInt(comfortSlider?.value ?? '5');
        const effectiveness = parseInt(effectivenessSlider?.value ?? '5');
        
        if (!rating || rating < 1) {
            this.showGothicNotification('Please select your overall sentiment first', 'error');
            return;
        }
        
        try {
            const response = await fetch('/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    rating,
                    comfort,
                    effectiveness,
                    emotion_state: this.emotionState
                })
            });
            
            if (response.ok) {
                this.showGothicNotification('Feedback submitted successfully', 'success');
                // Mark feedback toast time to suppress overlapping monitoring toast
                this._lastFeedbackToastAt = Date.now();
                // Ensure we remain subscribed to updates after feedback
                if (this.currentSession && this.currentSession.id && this.socket) {
                    this.socket.emit('start_emotion_monitoring', { session_id: this.currentSession.id });
                }
            } else {
                let errMsg = '';
                try {
                    const ct = response.headers.get('content-type') || '';
                    if (ct.includes('application/json')) {
                        const err = await response.json();
                        errMsg = err.error || err.message || response.statusText;
                    } else {
                        errMsg = await response.text();
                    }
                } catch (_) {
                    errMsg = response.statusText || 'Unknown error';
                }
                this.showGothicNotification(`Failed to submit feedback: ${errMsg}`, 'error');
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

    resetVisualizations() {
        // Clear trajectory canvas
        if (this.trajectoryCtx && this.trajectoryCanvas) {
            const c = this.trajectoryCanvas;
            this.trajectoryCtx.fillStyle = '#1a1a1a';
            this.trajectoryCtx.fillRect(0, 0, c.width, c.height);
            this.drawTrajectoryGrid();
        }
        // NEW: clear time-series canvas
        if (this.timeseriesCtx && this.timeseriesCanvas) {
            const c2 = this.timeseriesCanvas;
            this.timeseriesCtx.fillStyle = '#1a1a1a';
            this.timeseriesCtx.fillRect(0, 0, c2.width, c2.height);
            this.drawTimeSeriesAxes();
            this._tsHistory = [];
        }
        // Reset indicators
        const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
        ['faceV','faceA','faceC','faceUsed','voiceV','voiceA','voiceC','voiceUsed'].forEach(id => set(id, '-'));
        ['valenceValue','arousalValue','confidenceValue'].forEach(id => set(id, '-'));
        // Camera status back to Idle if not monitoring
        const cameraStatus = document.getElementById('cameraStatus');
        if (cameraStatus && !this.emotionMonitoring) {
            cameraStatus.textContent = 'Idle';
            cameraStatus.style.color = '#a0a0a0';
        }
    }
}

// NEW: Time-series canvas and rendering helpers
EmoTuneGothicClient.prototype.setupTimeSeriesCanvas = function() {
    this.timeseriesCanvas = document.getElementById('timeseriesCanvas');
    if (!this.timeseriesCanvas) return;
    this.timeseriesCtx = this.timeseriesCanvas.getContext('2d');
    this._resizeTimeSeriesCanvas();
    this.drawTimeSeriesAxes();
};

EmoTuneGothicClient.prototype._resizeTimeSeriesCanvas = function() {
    if (!this.timeseriesCanvas) return;
    const rect = this.timeseriesCanvas.getBoundingClientRect();
    this.timeseriesCanvas.width = rect.width;
    this.timeseriesCanvas.height = rect.height;
};

EmoTuneGothicClient.prototype.drawTimeSeriesAxes = function() {
    if (!this.timeseriesCtx || !this.timeseriesCanvas) return;
    const ctx = this.timeseriesCtx;
    const c = this.timeseriesCanvas;
    const w = c.width;
    const h = c.height;

    // Background
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, w, h);

    // Axes
    ctx.strokeStyle = '#404040';
    ctx.lineWidth = 1;

    // Y-grid for symmetric valence -1..1
    for (let i = -1; i <= 1; i += 0.5) {
        const y = ((1 - (i + 1) / 2)) * h;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
    }

    // Labels
    ctx.fillStyle = '#a0a0a0';
    ctx.font = '12px "Crimson Text", serif';
    ctx.textAlign = 'left';

    // Valence labels (left)
    const vTicks = [-1, -0.5, 0, 0.5, 1];
    for (const v of vTicks) {
        const y = ((1 - (v + 1) / 2)) * h;
        ctx.fillText(v.toFixed(1), 6, y - 2);
    }

    // Arousal labels (right) - symmetric
    ctx.textAlign = 'right';
    const aTicks = [-1, -0.5, 0, 0.5, 1];
    for (const a of aTicks) {
        const y = ((1 - (a + 1) / 2)) * h;
        ctx.fillText(a.toFixed(1), w - 6, y - 2);
    }

    // Title
    ctx.textAlign = 'center';
    ctx.fillText('Valence and Arousal vs Time', w / 2, 14);

    // Legend uses color only (red=valence, gold=arousal)
    ctx.fillStyle = '#8b0000';
    ctx.fillRect(w / 2 - 90, 22, 14, 3);
    ctx.fillStyle = '#a0a0a0';
    ctx.fillText('Valence', w / 2 - 60, 26);
    ctx.fillStyle = '#d4af37';
    ctx.fillRect(w / 2 + 20, 22, 14, 3);
    ctx.fillStyle = '#a0a0a0';
    ctx.fillText('Arousal', w / 2 + 54, 26);
};

EmoTuneGothicClient.prototype._updateTimeSeries = function(emotionState) {
    if (!emotionState) return;
    const ts = (typeof emotionState.timestamp === 'number') ? emotionState.timestamp * 1000 : Date.now();
    const v = typeof emotionState.valence === 'number' ? emotionState.valence : (emotionState.mean?.valence ?? 0);
    const a = typeof emotionState.arousal === 'number' ? emotionState.arousal : (emotionState.mean?.arousal ?? 0);

    this._tsHistory.push({ t: ts, v, a });

    // Prune by window seconds and max points
    const cutoff = Date.now() - this._tsWindowSeconds * 1000;
    while (this._tsHistory.length > 0 && this._tsHistory[0].t < cutoff) {
        this._tsHistory.shift();
    }
    if (this._tsHistory.length > this._tsMaxPoints) {
        this._tsHistory = this._tsHistory.slice(-this._tsMaxPoints);
    }

    // Throttle drawing
    const now = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
    if (now - this._lastTsDrawAt < this._tsUpdateIntervalMs) {
        if (!this._pendingTs) {
            this._pendingTs = true;
            requestAnimationFrame(() => {
                this._pendingTs = false;
                this.updateTimeSeriesVisualization();
            });
        }
        return;
    }
    this._lastTsDrawAt = now;
    this.updateTimeSeriesVisualization();
};

EmoTuneGothicClient.prototype.updateTimeSeriesVisualization = function(forceRedraw = false) {
    if (!this.timeseriesCtx || !this.timeseriesCanvas) return;
    const ctx = this.timeseriesCtx;
    const c = this.timeseriesCanvas;
    const w = c.width;
    const h = c.height;

    // Background and axes
    this.drawTimeSeriesAxes();

    if (this._tsHistory.length < 2) return;

    // Compute time window
    const tMin = this._tsHistory[0].t;
    const tMax = this._tsHistory[this._tsHistory.length - 1].t;
    const span = Math.max(1000, tMax - tMin);

    // Helpers to map
    const xFor = (t) => ((t - tMin) / span) * (w - 40) + 20; // padding 20px
    const yForValence = (v) => ((1 - (v + 1) / 2)) * (h - 40) + 20; // -1..1 -> 1..0, padded
    // UPDATED: normalize arousal that might arrive as 0..1
    const yForArousal = (a) => {
        const an = (a >= 0 && a <= 1) ? (a * 2 - 1) : a;
        return ((1 - (an + 1) / 2)) * (h - 40) + 20;
    };

    // Plot valence
    ctx.strokeStyle = '#8b0000';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < this._tsHistory.length; i++) {
        const p = this._tsHistory[i];
        const x = xFor(p.t);
        const y = yForValence(p.v);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Plot arousal
    ctx.strokeStyle = '#d4af37';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < this._tsHistory.length; i++) {
        const p = this._tsHistory[i];
        const x = xFor(p.t);
        const y = yForArousal(p.a);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // X-axis time labels (start, mid, end)
    ctx.fillStyle = '#a0a0a0';
    ctx.font = '12px "Crimson Text", serif';
    ctx.textAlign = 'center';
    const labels = [tMin, tMin + span / 2, tMax];
    labels.forEach((t) => {
        const x = xFor(t);
        const secsAgo = Math.max(0, Math.round((tMax - t) / 1000));
        ctx.fillText(`${secsAgo}s ago`, x, h - 6);
    });
};

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
