import numpy as np
from typing import Dict, List, Any
import math
from .parameter_space import MusicParameterSpace

class MusicRenderer:
    def __init__(self, param_space: MusicParameterSpace):
        self.param_space = param_space
        self.current_params = param_space.get_default_parameters()
        self.target_params = param_space.get_default_parameters()
        self.interpolation_steps = 10  # Steps for smooth transitions
        self.step_counter = 0
        
        # Musical structures
        self.scales = {
            'major': [0, 2, 4, 5, 7, 9, 11],
            'minor': [0, 2, 3, 5, 7, 8, 10],
            'dorian': [0, 2, 3, 5, 7, 9, 10],
            'blues': [0, 3, 5, 6, 7, 10]
        }
        
        self.chord_progressions = {
            'basic': ['I', 'V', 'vi', 'IV'],
            'jazz': ['IIMaj7', 'V7', 'IMaj7', 'vi7'],
            'modal': ['i', 'bVII', 'bVI', 'bVII']
        }

    def update_target_parameters(self, new_params: Dict[str, float]):
        """Set new target parameters for smooth interpolation"""
        self.target_params.update(new_params)
        self.step_counter = 0

    def interpolate_parameters(self) -> Dict[str, float]:
        """Smoothly interpolate current parameters toward targets"""
        if self.step_counter >= self.interpolation_steps:
            return self.current_params.copy()
        
        # Linear interpolation with easing
        alpha = self.step_counter / self.interpolation_steps
        alpha_eased = 0.5 * (1 - math.cos(alpha * math.pi))  # Cosine easing
        
        for param_name in self.current_params:
            if param_name in self.target_params:
                current = self.current_params[param_name]
                target = self.target_params[param_name]
                self.current_params[param_name] = current + alpha_eased * (target - current)
        
        self.step_counter += 1
        return self.current_params.copy()

    def render(self, params: Dict[str, float]) -> Dict[str, Any]:
        """Render a backend-agnostic music structure from parameters."""
        scale_type = self._select_scale(params)
        chord_prog = self._select_chord_progression(params)
        tempo = int(params.get('tempo_bpm', 120))
        music_struct = {
            'tempo': tempo,
            'scale': scale_type,
            'chord_progression': chord_prog,
            'melody': self._build_melody(params),
            'harmony': self._build_harmony(params, chord_prog),
            'rhythm': self._build_rhythm(params),
            'effects': self._build_effects(params)
        }
        return music_struct

    def _select_scale(self, params: Dict[str, float]) -> str:
        """Select scale based on valence and harmonic complexity"""
        dissonance = params.get('dissonance_level', 0.5)
        complexity = params.get('chord_complexity', 0.5)
        
        if dissonance > 0.7:
            return 'blues'
        elif complexity > 0.6:
            return 'dorian'
        elif dissonance < 0.3:
            return 'major'
        else:
            return 'minor'

    def _select_chord_progression(self, params: Dict[str, float]) -> List[str]:
        """Select chord progression based on parameters"""
        complexity = params.get('chord_complexity', 0.5)
        
        if complexity > 0.7:
            return self.chord_progressions['jazz']
        elif complexity > 0.4:
            return self.chord_progressions['modal']
        else:
            return self.chord_progressions['basic']

    def _build_melody(self, params: Dict[str, float]) -> Dict[str, Any]:
        """Build melody part of the music structure"""
        density = int(params.get('voice_density', 3))
        articulation = params.get('articulation', 0.5)
        brightness = params.get('brightness', 0.5)
        
        # Adjust octave based on brightness
        octave = 4 + int(brightness * 2)
        
        # Adjust note duration based on articulation
        dur = 0.25 if articulation > 0.7 else 0.5 if articulation > 0.3 else 1
        
        pattern = [0, 2, 4, 2]  # Example pattern
        
        return {
            'pattern': pattern,
            'octave': octave,
            'duration': dur,
            'density': density,
            'volume': params.get('overall_volume', 0.7)
        }

    def _build_harmony(self, params: Dict[str, float], progression: List[str]) -> Dict[str, Any]:
        """Build harmony part of the music structure"""
        volume = params.get('overall_volume', 0.7) * 0.6  # Harmony quieter than melody
        reverb = params.get('reverb_amount', 0.3)
        
        return {
            'progression': progression,
            'duration': 4,
            'volume': volume,
            'reverb': reverb
        }

    def _build_rhythm(self, params: Dict[str, float]) -> Dict[str, Any]:
        """Build rhythm part of the music structure"""
        complexity = params.get('rhythm_complexity', 0.5)
        volume = params.get('overall_volume', 0.7)
        
        if complexity > 0.7:
            pattern = [1, 0, 1, 0.5, 1, 0, 0.8, 0]
        elif complexity > 0.4:
            pattern = [1, 0, 1, 0, 1, 0, 1, 0]
        else:
            pattern = [1, 0, 0, 0, 1, 0, 0, 0]
        
        return {
            'pattern': pattern,
            'volume': volume
        }

    def _build_effects(self, params: Dict[str, float]) -> Dict[str, Any]:
        """Build effects part of the music structure"""
        reverb = params.get('reverb_amount', 0.3)
        filter_cutoff = params.get('filter_cutoff', 0.7)
        warmth = params.get('warmth', 0.6)
        
        return {
            'reverb': reverb,
            'filter_cutoff': filter_cutoff,
            'warmth': warmth
        }
