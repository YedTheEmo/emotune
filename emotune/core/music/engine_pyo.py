"""
Pyo-based Music Engine for Real-time Emotion-driven Music Generation
Replaces the problematic Sonic Pi approach with a robust, integrated solution.
"""

import time
import threading
from typing import Dict, Any, List, Optional
from collections import deque
import numpy as np
from emotune.utils.logging import get_logger

# Pyo imports (will be handled gracefully if not available)
try:
    from pyo import Server, Mixer, Sine, Freeverb, ButLP, Fader, Adsr, Pattern, midiToHz
    PYO_AVAILABLE = True
except Exception as e:
    PYO_AVAILABLE = False
    import traceback
    print("Pyo import failed:", e)
    traceback.print_exc()
    print("Warning: Pyo not available. Install with: pip install pyo")

from .engine_base import MusicEngineBase

class PyoMusicEngine(MusicEngineBase):
    """
    Real-time music generation engine using Pyo.
    Provides emotion-driven music synthesis without external dependencies.
    """
    
    def __init__(self, audio_backend="portaudio", sample_rate=44100, buffer_size=256):
        self.logger = get_logger()
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.audio_backend = audio_backend
        # Pyo server and components
        self.server = None
        self.mixer = None
        self.current_patterns = {}
        self.parameter_cache = {}
        self.pattern_player = None # Initialize pattern_player
        self.active_notes = [] # Use a simple list for manual management
        self.max_active_notes = 100 # Prevent too many simultaneous notes
        # --- Musical structure additions ---
        self.scale_notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C major scale (MIDI)
        self.rhythm_pattern = [0.5, 0.5, 1, 0.5, 0.5, 1, 0.5, 1]  # Rhythm pattern (seconds)
        self.scale_idx = 0
        self.chord_type = 'major'  # Will be set dynamically
        self.section_length = 8  # Number of notes per section
        self.polyphony = 2  # Number of simultaneous voices
        self.oscillators = []  # List of active oscillators
        self.envelopes = []    # List of active envelopes
        
        # Threading for real-time updates
        self.update_thread = None
        self.running = False
        self.update_queue = []
        self.update_lock = threading.Lock()
        
        # Initialize if Pyo is available
        if PYO_AVAILABLE:
            self._initialize_pyo()
        else:
            self.logger.error("Pyo not available. Music generation will be disabled.")
    
    def _initialize_pyo(self):
        """Initializes the Pyo server and mixer."""
        if self.server and self.server.getIsBooted():
            self.logger.info("Pyo server is already running.")
            return

        try:
            self.logger.info("Initializing Pyo server...")
            # Create Server object first, then boot, then start – avoids WinError 6.
            self.server = Server(sr=48000, winhost="wasapi", duplex=0)
            self.server.setVerbosity(0)
            self.server.boot()   # opens PortAudio stream
            self.server.start()  # starts processing
            
            # Allow a moment for the server to stabilize, especially with wasapi
            time.sleep(0.5)

            # Initialize mixer with a fixed voice pool and track assignments
            self._mixer_voices = 8
            self._mixer_voice_idx = 0
            self._voice_slots = [None] * self._mixer_voices  # (osc, env) tuples
            self.mixer = Mixer(outs=2, chnls=self._mixer_voices, time=0.025)
            # Initialize default amps for all voices (stereo)
            for v in range(self._mixer_voices):
                self.mixer.setAmp(v, 0, 0.5)
                self.mixer.setAmp(v, 1, 0.5)
            self.mixer.out()

            # ---------- SINGLE-OSCILLATOR MODEL ----------
            self.main_osc = None
            self.env = None
            
            # Initialize default parameters
            self._initialize_default_parameters()
            
            # Start update thread
            self.running = True
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()
            
            self.logger.info("Pyo music engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Pyo: {e}")
            self.server = None
            self.running = False
    
    def _initialize_default_parameters(self):
        """Initialize default musical parameters"""
        self.parameter_cache = {
            'tempo_bpm': 120.0,
            'overall_volume': 0.7,
            'voice_density': 2.0,
            'brightness': 0.5,
            'warmth': 0.5,
            'reverb_amount': 0.3,
            'articulation': 0.5,
            'rhythm_complexity': 0.5,
            'dissonance_level': 0.2,
            'chord_complexity': 0.5,
            'modulation_frequency': 0.1,
            'filter_cutoff': 0.7,
            'dynamic_range': 0.5,
            'crescendo_rate': 0.3,
            'accent_strength': 0.4
        }
    
    def _update_loop(self):
        """Background thread for processing parameter updates"""
        while self.running:
            try:
                with self.update_lock:
                    if self.update_queue:
                        params = self.update_queue.pop(0)
                        self._apply_parameters(params)
                
                time.sleep(0.01)  # 100Hz update rate
                
            except Exception as e:
                self.logger.error(f"Error in update loop: {e}")
                time.sleep(0.1)
    
    def _apply_parameters(self, params: Dict[str, float]):
        """Apply parameters to the music engine with proper type conversion"""
        if not PYO_AVAILABLE:
            self.logger.warning("Pyo not available - cannot apply parameters")
            return
        
        try:
            # FIXED: Convert all numpy types to Python types
            converted_params = {}
            for key, value in params.items():
                if isinstance(value, np.ndarray):
                    converted_params[key] = float(value.item() if value.size == 1 else value[0])
                elif isinstance(value, (np.integer, np.floating)):
                    converted_params[key] = float(value)
                else:
                    converted_params[key] = value
            
            # Apply tempo changes
            if 'tempo_bpm' in converted_params:
                self._update_tempo(float(converted_params['tempo_bpm']))
            
            # Apply volume changes
            if 'overall_volume' in converted_params:
                self._update_volume(float(converted_params['overall_volume']))
            
            # Apply synthesis parameters
            self._update_synthesis_parameters(converted_params)
            
            # Apply effects
            self._update_effects(converted_params)
            
            self.logger.debug(f"Successfully applied {len(converted_params)} parameters")
            
        except Exception as e:
            self.logger.error(f"Error applying parameters: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _update_tempo(self, tempo_bpm: float):
        """Update musical tempo"""
        self.parameter_cache['tempo_bpm'] = max(40.0, min(200.0, tempo_bpm))
    
    def _update_volume(self, volume: float):
        """Update overall volume"""
        if self.mixer:
            vol = max(0.0, min(1.0, volume))
            voices = getattr(self, '_mixer_voices', 1)
            for v in range(voices):
                # set stereo amps per voice
                try:
                    self.mixer.setAmp(v, 0, vol)
                    self.mixer.setAmp(v, 1, vol)
                except Exception:
                    # best-effort if voice index invalid
                    pass
    
    def _update_synthesis_parameters(self, params: Dict[str, float]):
        """Update synthesis parameters with proper type conversion"""
        try:
            # FIXED: Convert all numpy types to Python types
            converted_params = {}
            for key, value in params.items():
                if isinstance(value, np.ndarray):
                    converted_params[key] = float(value.item() if value.size == 1 else value[0])
                elif isinstance(value, (np.integer, np.floating)):
                    converted_params[key] = float(value)
                else:
                    converted_params[key] = value
            
            # Update synthesis parameters
            for param_name, value in converted_params.items():
                if param_name in ['brightness', 'warmth', 'roughness']:
                    # These are synthesis parameters that affect the sound
                    self.logger.debug(f"Updated synthesis parameter {param_name}: {value}")
                    
        except Exception as e:
            self.logger.error(f"Error updating synthesis parameters: {e}")
    
    def _update_effects(self, params: Dict[str, float]):
        """Update audio effects with proper type conversion"""
        try:
            # FIXED: Convert all numpy types to Python types
            converted_params = {}
            for key, value in params.items():
                if isinstance(value, np.ndarray):
                    converted_params[key] = float(value.item() if value.size == 1 else value[0])
                elif isinstance(value, (np.integer, np.floating)):
                    converted_params[key] = float(value)
                else:
                    converted_params[key] = value
            
            # Update effects
            for param_name, value in converted_params.items():
                if param_name in ['reverb_amount', 'filter_cutoff']:
                    # These are effect parameters
                    self.logger.debug(f"Updated effect parameter {param_name}: {value}")
                    
        except Exception as e:
            self.logger.error(f"Error updating effects: {e}")
    
    def _create_emotion_pattern(self, valence: float, arousal: float) -> Dict[str, Any]:
        """Create musical pattern based on emotion"""
        pattern = {
            'base_frequency': 220 + (valence + 1) * 110,  # 220-440 Hz
            'tempo_bpm': max(40, min(160, 80 + arousal * 80)),  # 40-160 BPM with bounds
            'brightness': max(0, min(1, (valence + 1) / 2)),  # 0-1 with bounds
            'warmth': max(0, min(1, (1 - arousal) / 2)),  # 0-1 with bounds
            'rhythm_complexity': max(0, min(1, arousal)),  # 0-1 with bounds
            'dissonance_level': max(0, min(1, (1 - valence) / 2)),  # 0-1 with bounds
            'voice_density': max(1, min(3, 1 + arousal * 2)),  # 1-3 voices with bounds
            'reverb_amount': max(0.1, min(0.8, 0.2 + arousal * 0.3)),  # 0.1-0.8 with bounds
            'articulation': max(0.1, min(1, 0.3 + valence * 0.4)),  # 0.1-1 with bounds
        }
        return pattern
    
    def update_parameters(self, params: Dict[str, float]):
        """Update music parameters in real time"""
        if not PYO_AVAILABLE:
            self.logger.warning("Pyo not available - cannot update parameters")
            return
        
        # Queue parameter update for thread-safe processing
        with self.update_lock:
            self.update_queue.append(params.copy())
        
        self.logger.debug(f"Queued parameter update: {params}")
    
    def play(self, music_struct: Dict[str, Any]):
        """Play music based on provided parameters"""
        if not PYO_AVAILABLE:
            self.logger.warning("Pyo not available - cannot play music")
            return
        
        try:
            # Use the parameters provided by the mapping engine instead of generating our own
            parameters = music_struct.get('parameters', {})
            emotion_data = music_struct.get('emotion', {})
            
            if not parameters:
                self.logger.warning("No parameters provided to Pyo engine")
                return
            
            # Convert the provided parameters to Pyo-compatible format
            pattern = self._convert_parameters_to_pattern(parameters, emotion_data)
            
            # THE FIX: Only generate a new musical pattern if one is not already playing.
            # Otherwise, the existing Pattern will pick up the new emotion data
            # from `self._latest_pattern` on its next note.
            if not self.pattern_player or not self.pattern_player.isPlaying():
                self._generate_musical_content(pattern)
            
            valence = emotion_data.get('mean', {}).get('valence', emotion_data.get('valence', 0.0))
            arousal = emotion_data.get('mean', {}).get('arousal', emotion_data.get('arousal', 0.0))
            
            self.logger.info(f"Playing music with {len(parameters)} parameters: valence={valence:.2f}, arousal={arousal:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error playing music: {e}")
    
    def pause(self):
        """Pause music playback with a fade-out."""
        if self.mixer:
            self.mixer.stop()
            self.logger.info("Music paused.")

    def regenerate(self):
        """Regenerate music with the latest parameters."""
        self.logger.info("Regenerating music...")
        if hasattr(self, '_latest_pattern'):
            self._generate_musical_content(self._latest_pattern)

    def _convert_parameters_to_pattern(self, parameters: Dict[str, float], emotion_data: Dict) -> Dict[str, Any]:
        """Convert provided music parameters to Pyo-compatible pattern"""
        # Extract valence/arousal for base frequency calculation
        if 'mean' in emotion_data:
            valence = emotion_data['mean'].get('valence', 0.0)
            arousal = emotion_data['mean'].get('arousal', 0.0)
        else:
            valence = emotion_data.get('valence', 0.0)
            arousal = emotion_data.get('arousal', 0.0)
        
        # Use the provided parameters directly - FIXED: Don't override with defaults
        pattern = {
            'base_frequency': float(220 + (valence + 1) * 110),  # Calculate from emotion
            'tempo_bpm': float(parameters.get('tempo_bpm', 120.0)),
            'brightness': float(parameters.get('brightness', 0.5)),
            'warmth': float(parameters.get('warmth', 0.5)),
            'rhythm_complexity': float(parameters.get('rhythm_complexity', 0.5)),
            'dissonance_level': float(parameters.get('dissonance_level', 0.2)),
            'voice_density': float(parameters.get('voice_density', 2.0)),
            'reverb_amount': float(parameters.get('reverb_amount', 0.3)),
            'articulation': float(parameters.get('articulation', 0.5)),
            'overall_volume': float(parameters.get('overall_volume', 0.7)),
        }
        
        self.logger.debug(f"Converted parameters to pattern: tempo={pattern['tempo_bpm']}, voices={pattern['voice_density']}")
        self._latest_pattern = pattern # Cache the latest pattern
        return pattern
    
    def _generate_musical_content(self, pattern: Dict[str, Any]):
        """Generate musical content using scale-based melody, chord progression, and rhythm."""
        if not self.server or not self.running:
            return
        # Stop previous pattern if running
        if self.pattern_player and self.pattern_player.isPlaying():
            self.pattern_player.stop()
        # Set musical parameters
        tempo_bpm = pattern.get('tempo_bpm', 120)
        # Derive rhythmic subdivision from arousal (higher arousal -> shorter steps)
        arousal = float(pattern.get('arousal', 0.0)) if 'arousal' in pattern else 0.0
        base_step = 0.25 if arousal > 0.5 else 0.5
        self.rhythm_pattern = [base_step] * 4 + [base_step * 2, base_step, base_step * 2, base_step]
        self.section_length = int(pattern.get('section_length', 8)) if 'section_length' in pattern else 8
        self.polyphony = int(pattern.get('voice_density', 2))
        # Choose scale and chord type based on valence
        valence = float(pattern.get('valence', 0)) if 'valence' in pattern else 0.0
        if valence > 0:
            # Positive: brighter key
            self.scale_notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C major
            self.chord_type = 'major'
        else:
            # Negative: minor mode
            self.scale_notes = [57, 59, 60, 62, 64, 65, 67, 69]  # A minor
            self.chord_type = 'minor'
        # Clear previous oscillators/envelopes
        for osc in self.oscillators:
            if hasattr(osc, 'stop'): osc.stop()
        self.oscillators = []
        self.envelopes = []
        # Start pattern
        self._section_counter = 0
        self.pattern_player = Pattern(self._play_note, time=self.rhythm_pattern[0]).play()

    def _play_note(self):
        """
        Play a note from the scale, using a chord for harmony and polyphony.
        Uses Adsr envelope and Sine oscillator (PYO 1.05 verified).
        """
        try:
            # PYO 1.0.5 SAFETY: Skip if server not booted or engine not running
            if not self.running or self.server is None:
                return
            try:
                if hasattr(self.server, 'getIsBooted') and not self.server.getIsBooted():
                    return
            except Exception:
                # If API missing, proceed cautiously
                pass
            if self.mixer is None:
                return
            # Change key/progression every section_length notes
            if self._section_counter % max(1, self.section_length) == 0 and self._section_counter > 0:
                # Rotate scale root to introduce movement
                root_shift = int(np.random.choice([-2, 0, 2]))  # step down, same, or up
                self.scale_notes = [n + root_shift for n in self.scale_notes]
            # Get current note and chord
            note = self.scale_notes[self.scale_idx % len(self.scale_notes)]
            chord = self._generate_chord(midiToHz(note), self._latest_pattern if hasattr(self, '_latest_pattern') else {})
            # Polyphony: play up to self.polyphony notes from chord
            for i in range(self.polyphony):
                freq = chord[i % len(chord)] if chord else midiToHz(note)
                env = Adsr(attack=0.01, decay=0.12, sustain=0.65, release=0.25, dur=self.rhythm_pattern[self.scale_idx % len(self.rhythm_pattern)], mul=0.25).play()
                osc = Sine(freq=freq, mul=env)
                # Assign to a mixer voice with cleanup and reuse
                vidx = getattr(self, '_mixer_voice_idx', 0)
                voices = getattr(self, '_mixer_voices', 1)
                # If a previous osc is assigned on this voice, remove and stop it
                prev = None
                try:
                    prev = self._voice_slots[vidx]
                except Exception:
                    prev = None
                if prev is not None:
                    try:
                        p_osc, p_env = prev
                        if hasattr(p_env, 'stop'):
                            p_env.stop()
                        if hasattr(p_osc, 'stop'):
                            p_osc.stop()
                    except Exception:
                        pass
                    try:
                        self.mixer.delInput(vidx)
                    except Exception:
                        pass
                # Add new osc to mixer voice
                try:
                    self.mixer.addInput(vidx, osc)
                    # Ensure amps present for this voice using current volume
                    try:
                        vol = float(self.parameter_cache.get('overall_volume', 0.5))
                    except Exception:
                        vol = 0.5
                    self.mixer.setAmp(vidx, 0, vol)
                    self.mixer.setAmp(vidx, 1, vol)
                    self._voice_slots[vidx] = (osc, env)
                except Exception as e:
                    self.logger.debug(f"Mixer addInput failed: {e}")
                    # If mixer rejects, stop objects to avoid leaks
                    try:
                        env.stop()
                        osc.stop()
                    except Exception:
                        pass
                    self._voice_slots[vidx] = None
                # advance voice index
                self._mixer_voice_idx = (vidx + 1) % max(1, voices)
                self.oscillators.append(osc)
                self.envelopes.append(env)
            # Cleanup old oscillators/envelopes
            if len(self.oscillators) > self.polyphony * self.section_length:
                for osc in self.oscillators[:-self.polyphony * self.section_length]:
                    if hasattr(osc, 'stop'): osc.stop()
                self.oscillators = self.oscillators[-self.polyphony * self.section_length:]
            if len(self.envelopes) > self.polyphony * self.section_length:
                for env in self.envelopes[:-self.polyphony * self.section_length]:
                    if hasattr(env, 'stop'): env.stop()
                self.envelopes = self.envelopes[-self.polyphony * self.section_length:]
            # Advance indices
            self.scale_idx += 1
            self._section_counter += 1
            # Update pattern timing for next note
            next_time = self.rhythm_pattern[self.scale_idx % len(self.rhythm_pattern)]
            if self.pattern_player:
                self.pattern_player.time = next_time
        except Exception as e:
            self.logger.error(f"Error in _play_note: {e}")

    def _get_next_note(self):
        """Get the next note in the sequence based on the current emotion."""
        base_freq = self._latest_pattern.get('base_frequency', 220)
        chord = self._generate_chord(base_freq, self._latest_pattern)
        # CRITICAL FIX: Cast the numpy.float64 from np.random.choice to a standard Python float.
        return float(np.random.choice(chord))
    
    def _update_existing_patterns(self, pattern: Dict[str, Any]):
        """Update existing patterns instead of creating new ones"""
        try:
            base_freq = pattern.get('base_frequency', 440.0)
            density = max(1, min(4, int(pattern.get('voice_density', 2))))
            
            # Generate new chord
            chord_notes = self._generate_chord(base_freq, pattern)
            
            # Update existing oscillators
            for i, (pattern_id, oscillator) in enumerate(self.current_patterns.items()):
                if i < len(chord_notes) and i < density:
                    try:
                        new_freq = float(chord_notes[i])
                        if 20 <= new_freq <= 20000:
                            oscillator.freq = new_freq
                            self.logger.debug(f"Updated oscillator {pattern_id} to {new_freq:.1f} Hz")
                    except Exception as e:
                        self.logger.warning(f"Failed to update oscillator {pattern_id}: {e}")
            
        except Exception as e:
            self.logger.error(f"Error updating existing patterns: {e}")
    
    def _create_new_patterns(self, pattern: Dict[str, Any]):
        """Create new patterns only when none exist"""
        try:
            # Create harmonic content
            base_freq = pattern.get('base_frequency', 440.0)
            density = max(1, min(4, int(pattern.get('voice_density', 2))))
            
            # FIXED: Clear old patterns before creating new ones
            self._clear_patterns()
            
            # Generate chord based on emotion
            chord_notes = self._generate_chord(base_freq, pattern)
            
            # Create oscillators for each note with better error handling
            successful_oscs = 0
            for i, note_freq in enumerate(chord_notes[:density]):
                try:
                    # Validate frequency
                    if not (20 <= note_freq <= 20000):
                        self.logger.warning(f"Invalid frequency {note_freq}, skipping")
                        continue
                    
                    # Create oscillator with safer parameters
                    volume = min(0.1, 0.3/max(1, density))  # Lower volume to prevent clipping
                    
                    # FIXED: Convert numpy types to Python types
                    freq = float(note_freq)
                    vol = float(volume)
                    
                    oscillator = Sine(freq=freq, mul=vol)
                    
                    # FIXED: Add oscillator to the mixer
                    self.mixer.addInput(0, oscillator)
                    
                    # Store oscillator
                    pattern_id = f"osc_{i}_{int(time.time())}"
                    self.current_patterns[pattern_id] = oscillator
                    successful_oscs += 1
                    
                except Exception as e:
                    self.logger.error(f"Error creating oscillator {i}: {e}")
            
            self.logger.debug(f"Successfully created {successful_oscs} oscillators")
            
            # Add effects
            self._add_effects(pattern)
            
        except Exception as e:
            self.logger.error(f"Error creating new patterns: {e}")
    
    def _add_effects(self, pattern: Dict[str, Any]):
        """Add audio effects to the current patterns"""
        try:
            # Add reverb
            reverb_amount = float(pattern.get('reverb_amount', 0.3))
            if reverb_amount > 0 and hasattr(self, 'server') and self.current_patterns:
                try:
                    # FIXED: Create reverb with proper input source
                    # Use the first oscillator as input source
                    first_osc = list(self.current_patterns.values())[0]
                    self.reverb = Freeverb(first_osc, mul=reverb_amount)
                    # FIXED: Don't use addInput since Mixer doesn't have that method
                    # The reverb will be heard through the server
                except Exception as e:
                    self.logger.debug(f"Reverb failed, using dry signal: {e}")
            
            # Add filter
            filter_cutoff = float(pattern.get('filter_cutoff', 0.6))
            if filter_cutoff > 0 and hasattr(self, 'server') and self.current_patterns:
                try:
                    # FIXED: Create filter with proper input source
                    # Use the first oscillator as input source
                    first_osc = list(self.current_patterns.values())[0]
                    cutoff = filter_cutoff * 2000 + 200  # 200-2200 Hz
                    self.filter = ButLP(first_osc, freq=float(cutoff))
                    # FIXED: Don't use addInput since Mixer doesn't have that method
                    # The filter will be heard through the server
                except Exception as e:
                    self.logger.debug(f"Filter failed: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error adding effects: {e}")
    
    def _apply_mixer_effects(self, pattern: Dict[str, Any]):
        """Apply audio effects to the main mixer."""
        try:
            # Apply reverb
            reverb_amount = float(pattern.get('reverb_amount', 0.3))
            if reverb_amount > 0 and hasattr(self, 'server'):
                if not hasattr(self, 'reverb') or not self.reverb:
                    self.reverb = Freeverb(self.mixer, mul=reverb_amount).out()
                else:
                    self.reverb.mul = reverb_amount
            
            # Apply filter
            filter_cutoff = float(pattern.get('filter_cutoff', 0.6))
            if filter_cutoff > 0 and hasattr(self, 'server'):
                cutoff_freq = filter_cutoff * 2000 + 200  # 200-2200 Hz
                if not hasattr(self, 'filter') or not self.filter:
                    self.filter = ButLP(self.reverb or self.mixer, freq=float(cutoff_freq)).out()
                else:
                    self.filter.freq = float(cutoff_freq)

        except Exception as e:
            self.logger.error(f"Error applying mixer effects: {e}")

    def _clear_old_patterns(self):
        """Clear old patterns to prevent audio buildup"""
        try:
            patterns_to_remove = []
            current_time = time.time()
            
            for pattern_id, osc in self.current_patterns.items():
                # Remove patterns older than 10 seconds
                try:
                    pattern_time = int(pattern_id.split('_')[-1])
                    if current_time - pattern_time > 10:
                        patterns_to_remove.append(pattern_id)
                except:
                    patterns_to_remove.append(pattern_id)
            
            for pattern_id in patterns_to_remove:
                try:
                    osc = self.current_patterns.pop(pattern_id)
                    if hasattr(osc, 'stop'):
                        osc.stop()
                except Exception as e:
                    self.logger.debug(f"Error clearing pattern {pattern_id}: {e}")
                    
        except Exception as e:
            self.logger.debug(f"Error clearing old patterns: {e}")
    
    def _generate_chord(self, base_freq: float, pattern: Dict[str, Any]) -> List[float]:
        """Generate chord frequencies based on emotion"""
        # Major chord for positive valence, minor for negative
        if pattern.get('valence', 0) > 0:
            # Major chord (1, 5/4, 3/2)
            ratios = [1.0, 1.25, 1.5]
        else:
            # Minor chord (1, 6/5, 3/2)
            ratios = [1.0, 1.2, 1.5]
        
        # Add dissonance
        dissonance = pattern.get('dissonance_level', 0)
        if dissonance > 0.3:
            # Add tritone (diminished fifth)
            ratios.append(1.414)  # √2
        
        return [base_freq * ratio for ratio in ratios]
    
    def stop(self):
        """Stop all music playback"""
        if not PYO_AVAILABLE:
            return
        
        try:
            # Stop update thread first
            self.running = False
            if self.update_thread and self.update_thread.is_alive():
                self.update_thread.join(timeout=1.0)
            # PYO 1.0.5: Stop pattern before any server/mixer teardown to prevent callbacks
            try:
                if self.pattern_player and hasattr(self.pattern_player, 'isPlaying') and self.pattern_player.isPlaying():
                    self.pattern_player.stop()
            except Exception:
                pass
            self.pattern_player = None
            
            # FIXED: Use a fader for smooth fade-out with correct API.
            if self.mixer and self.server and (not hasattr(self.server, 'getIsBooted') or self.server.getIsBooted()):
                # The old getMul()/getAdd() are deprecated. Access attributes directly.
                fader = Fader(fadeout=0.5, mul=self.mixer.mul, add=self.mixer.add).play()
                time.sleep(0.5)
            
            # Stop all patterns
            self._clear_patterns()
            
            # Stop server with proper cleanup
            if self.server:
                try:
                    # Check if server is still running before stopping
                    if hasattr(self.server, 'getIsBooted') and self.server.getIsBooted():
                        self.server.stop()
                        time.sleep(0.1)  # Give time for cleanup
                    self.server.shutdown()
                except Exception as e:
                    self.logger.debug(f"Error during server shutdown: {e}")
                finally:
                    self.server = None
            
            self.logger.info("Pyo music engine stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping music engine: {e}")
    
    def _clear_patterns(self):
        """Stop and clear all current oscillators."""
        if not self.current_patterns:
            # Even if no patterns, ensure mixer voices cleared
            if getattr(self, '_voice_slots', None) is not None:
                for vidx, slot in enumerate(self._voice_slots):
                    try:
                        if slot:
                            o, e = slot
                            if hasattr(e, 'stop'): e.stop()
                            if hasattr(o, 'stop'): o.stop()
                    except Exception:
                        pass
                    try:
                        self.mixer.delInput(vidx)
                    except Exception:
                        pass
                self._voice_slots = [None] * getattr(self, '_mixer_voices', 0)
            return
        
        for osc in self.current_patterns.values():
            if hasattr(osc, 'stop'):
                osc.stop()
        self.current_patterns.clear()
        self.active_notes.clear() # Also clear the active_notes list
        # Guard mixer operations for Pyo 1.0.5 compatibility: remove inputs per voice
        if self.mixer:
            try:
                for vidx, slot in enumerate(self._voice_slots):
                    try:
                        if slot:
                            o, e = slot
                            if hasattr(e, 'stop'): e.stop()
                            if hasattr(o, 'stop'): o.stop()
                    except Exception:
                        pass
                    try:
                        self.mixer.delInput(vidx)
                    except Exception:
                        pass
                self._voice_slots = [None] * getattr(self, '_mixer_voices', 0)
                # keep mixer routed
                if self.server and (not hasattr(self.server, 'getIsBooted') or self.server.getIsBooted()):
                    self.mixer.out()
            except Exception as e:
                self.logger.debug(f"Mixer clear/out failed: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current engine status"""
        return {
            'available': PYO_AVAILABLE,
            'running': self.running and self.server is not None,
            'active_patterns': len(self.current_patterns),
            'current_parameters': self.parameter_cache.copy(),
            'audio_backend': self.audio_backend,
            'sample_rate': self.sample_rate
        }
    
    def __del__(self):
        """Cleanup on destruction"""
        self.stop()


# Fallback engine for when Pyo is not available
class FallbackMusicEngine(MusicEngineBase):
    """Fallback music engine that logs parameters without generating sound"""
    
    def __init__(self):
        self.logger = get_logger()
        self.current_parameters = {}
    
    def update_parameters(self, params: Dict[str, float]):
        """Log parameter updates"""
        self.current_parameters.update(params)
        self.logger.info(f"Music parameters (fallback): {params}")
    
    def play(self, music_struct: Dict[str, Any]):
        """Log music play request"""
        self.logger.info(f"Music play request (fallback): {music_struct}")
    
    def stop(self):
        """Log stop request"""
        self.logger.info("Music stop request (fallback)")
    
    def get_status(self) -> Dict[str, Any]:
        """Get fallback status"""
        return {
            'available': False,
            'running': False,
            'current_parameters': self.current_parameters.copy(),
            'fallback_mode': True
        }


# Factory function to create appropriate engine
def create_music_engine(engine_type: str = "auto") -> MusicEngineBase:
    """
    Create music engine based on availability and preference.
    
    Args:
        engine_type: "pyo", "fallback", or "auto"
    
    Returns:
        MusicEngineBase instance
    """
    if engine_type == "pyo" or (engine_type == "auto" and PYO_AVAILABLE):
        return PyoMusicEngine()
    else:
        return FallbackMusicEngine()
