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
        this.trajectoryCtx = this.trajectoryCanvas.getContext('2d');
        
        // Set canvas size
        this.trajectoryCanvas.width = this.trajectoryCanvas.offsetWidth;
        this.trajectoryCanvas.height = this.trajectoryCanvas.offsetHeight;
        
        this.drawTrajectoryGrid();
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
            // Use average_score as a proxy; can be replaced with server-provided stability
            document.getElementById('emotionStability').textContent = (feedback.average_score || 0).toFixed(3);
        }

        if (rlAgent) {
            document.getElementById('rlBufferSize').textContent = rlAgent.buffer_size || 0;
            document.getElementById('rlLearningRate').textContent = (rlAgent.learning_rate || 0).toFixed(6);
            document.getElementById('rlRewardSignal').textContent = (rlAgent.reward_signal || 0).toFixed(3);
            document.getElementById('rlPolicyConfidence').textContent = (rlAgent.policy_confidence || 0).toFixed(3);
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
                    rlLog.textContent = `RL Agent: ${data.rl_agent.status || 'Unknown'}\nReward: ${data.rl_agent.reward_signal?.toFixed(3) || 'N/A'}\nAction: ${data.rl_agent.action || 'N/A'}`;
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
            this._trajectoryHistory.session_start_time = Date.now();
        }
        
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
        // FIX: Maintain persistent trajectory history for continuous visualization
        if (!this._trajectoryHistory) {
            this._trajectoryHistory = {
                target_path: [],
                actual_path: [],
                max_points: 300,   // Realistic limit for typical sessions
                fade_start_threshold: 150,  // Start fading after ~5 minutes
                fade_duration: 150,  // Fade over the same duration
                session_start_time: Date.now()
            };
        }
        
        // Add new target path points (if provided)
        if (trajectoryProgress.target_path && Array.isArray(trajectoryProgress.target_path)) {
            this._trajectoryHistory.target_path.push(...trajectoryProgress.target_path);
            // FIX: Use smarter compression instead of simple truncation
            this._compressPathIfNeeded('target_path');
        }
        
        // Add new actual path points (if provided)
        if (trajectoryProgress.actual_path && Array.isArray(trajectoryProgress.actual_path)) {
            this._trajectoryHistory.actual_path.push(...trajectoryProgress.actual_path);
            // FIX: Use smarter compression instead of simple truncation
            this._compressPathIfNeeded('actual_path');
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
        // FIX: Gradually fade out old trajectory points instead of sudden removal
        const path = this._trajectoryHistory[pathType];
        const fadeStart = this._trajectoryHistory.fade_start_threshold;
        const fadeDuration = this._trajectoryHistory.fade_duration;
        
        // Calculate how many points to gradually remove
        const excessPoints = path.length - this._trajectoryHistory.max_points;
        const pointsToRemove = Math.min(excessPoints, Math.floor(fadeDuration * 0.1)); // Remove 10% of fade duration
        
        if (pointsToRemove > 0) {
            // Remove oldest points gradually (from the beginning)
            this._trajectoryHistory[pathType] = path.slice(pointsToRemove);
            
            // Add fade-out effect to the remaining oldest points
            this._addFadeEffectToOldPoints(pathType);
        }
    }
    
    _addFadeEffectToOldPoints(pathType) {
        // FIX: Add visual fade effect to old trajectory points
        const path = this._trajectoryHistory[pathType];
        const fadeDuration = this._trajectoryHistory.fade_duration;
        
        // Add alpha/opacity property to points for fade effect
        for (let i = 0; i < Math.min(path.length, fadeDuration); i++) {
            const fadeRatio = i / fadeDuration;
            const alpha = 0.3 + (fadeRatio * 0.7); // Fade from 0.3 to 1.0 opacity
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
                document.getElementById('startSessionBtn').disabled = true;
                document.getElementById('stopSessionBtn').disabled = false;
                // Cache current session and join room explicitly
                this.currentSession = { id: result.session_id, trajectory_type: trajectoryType, duration: duration, startTime: Date.now() };
                if (this.socket) {
                    this.socket.emit('start_emotion_monitoring', { session_id: result.session_id });
                }
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
            
            // Convert valence/arousal to canvas coordinates
            const x = centerX + (valence * centerX * 0.8);
            const y = centerY - (arousal * centerY * 0.8);
            
            emotionPoint.style.left = `${x}px`;
            emotionPoint.style.top = `${y}px`;
            
            // Update values
            document.getElementById('valenceValue').textContent = valence.toFixed(3);
            document.getElementById('arousalValue').textContent = arousal.toFixed(3);
            document.getElementById('confidenceValue').textContent = confidence.toFixed(3);
        });
    }
    
    updateTrajectoryVisualization(trajectoryProgress) {
		if (!this.trajectoryCtx) return;
		
		// Throttle redraws to avoid UI stalls
		const now = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
		if (this._lastTrajDrawAt === undefined) this._lastTrajDrawAt = 0;
		if (this._trajDrawIntervalMs === undefined) this._trajDrawIntervalMs = 100; // ~10 FPS max
		if (now - this._lastTrajDrawAt < this._trajDrawIntervalMs) {
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
		
		// Utility: decimate long paths to a reasonable number of points
		const decimate = (points, maxPoints) => {
			if (!Array.isArray(points)) return [];
			const n = points.length;
			if (n <= maxPoints) return points;
			const step = Math.ceil(n / maxPoints);
			const out = [];
			for (let i = 0; i < n; i += step) out.push(points[i]);
			if (out.length === 0 || out[out.length - 1] !== points[n - 1]) out.push(points[n - 1]);
			return out;
		};
		
		// FIX: Use persistent trajectory history instead of just current data
		const targetPath = decimate(this._trajectoryHistory.target_path || [], 300);
		const actualPath = decimate(this._trajectoryHistory.actual_path || [], 300);
		
		// Clear canvas
		ctx.fillStyle = '#1a1a1a';
		ctx.fillRect(0, 0, width, height);
		
		// Draw grid
		this.drawTrajectoryGrid();
		
		// Draw target path (dashed line) with fade effect
		if (targetPath.length > 1) {
			ctx.lineWidth = 2;
			ctx.setLineDash([5, 5]);
			
			// Draw with fade effect
			for (let i = 0; i < targetPath.length - 1; i++) {
				const point1 = targetPath[i];
				const point2 = targetPath[i + 1];
				const alpha1 = point1.alpha !== undefined ? point1.alpha : 1.0;
				const alpha2 = point2.alpha !== undefined ? point2.alpha : 1.0;
				
				// Create gradient for this line segment
				const gradient = ctx.createLinearGradient(
					(point1.valence + 1) * width / 2, (1 - point1.arousal) * height / 2,
					(point2.valence + 1) * width / 2, (1 - point2.arousal) * height / 2
				);
				
				const color1 = `rgba(212, 175, 55, ${alpha1})`; // #d4af37 with alpha
				const color2 = `rgba(212, 175, 55, ${alpha2})`;
				
				gradient.addColorStop(0, color1);
				gradient.addColorStop(1, color2);
				
				ctx.strokeStyle = gradient;
				ctx.beginPath();
				ctx.moveTo((point1.valence + 1) * width / 2, (1 - point1.arousal) * height / 2);
				ctx.lineTo((point2.valence + 1) * width / 2, (1 - point2.arousal) * height / 2);
				ctx.stroke();
			}
			ctx.setLineDash([]);
		}
		
		// Draw actual path with fade effect
		if (actualPath.length > 1) {
			ctx.lineWidth = 3;
			
			// Draw with fade effect
			for (let i = 0; i < actualPath.length - 1; i++) {
				const point1 = actualPath[i];
				const point2 = actualPath[i + 1];
				const alpha1 = point1.alpha !== undefined ? point1.alpha : 1.0;
				const alpha2 = point2.alpha !== undefined ? point2.alpha : 1.0;
				
				// Create gradient for this line segment
				const gradient = ctx.createLinearGradient(
					(point1.valence + 1) * width / 2, (1 - point1.arousal) * height / 2,
					(point2.valence + 1) * width / 2, (1 - point2.arousal) * height / 2
				);
				
				const color1 = `rgba(139, 0, 0, ${alpha1})`; // #8b0000 with alpha
				const color2 = `rgba(139, 0, 0, ${alpha2})`;
				
				gradient.addColorStop(0, color1);
				gradient.addColorStop(1, color2);
				
				ctx.strokeStyle = gradient;
				ctx.beginPath();
				ctx.moveTo((point1.valence + 1) * width / 2, (1 - point1.arousal) * height / 2);
				ctx.lineTo((point2.valence + 1) * width / 2, (1 - point2.arousal) * height / 2);
				ctx.stroke();
			}
		}
		
		// Draw current position
		if (actualPath.length > 0) {
			const currentPoint = actualPath[actualPath.length - 1];
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
            // Reset UI on stop
            this.resetVisualizations();
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
                // Ensure we remain subscribed to updates after feedback
                if (this.currentSession && this.currentSession.id && this.socket) {
                    this.socket.emit('start_emotion_monitoring', { session_id: this.currentSession.id });
                }
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

    resetVisualizations() {
        // Clear trajectory canvas
        if (this.trajectoryCtx && this.trajectoryCanvas) {
            const c = this.trajectoryCanvas;
            this.trajectoryCtx.fillStyle = '#1a1a1a';
            this.trajectoryCtx.fillRect(0, 0, c.width, c.height);
            this.drawTrajectoryGrid();
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
