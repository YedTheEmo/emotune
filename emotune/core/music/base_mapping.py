import numpy as np
from typing import Dict, Tuple
from .parameter_space import MusicParameterSpace

class BaseMappingEngine:
    def __init__(self):
        self.param_space = MusicParameterSpace()
        self._init_mapping_functions()
    
    def _init_mapping_functions(self):
        """Initialize base mapping functions from valence-arousal to music parameters"""
        # These are research-based mappings that can be overridden by RL
        self.mappings = {
            # Tempo mappings
            "tempo_bpm": lambda v, a: 90 + 60 * a,  # Higher arousal = faster tempo
            "tempo_variation": lambda v, a: 0.05 + 0.2 * (1 - v),  # Lower valence = more variation
            "rhythm_complexity": lambda v, a: 0.3 + 0.4 * a,  # Higher arousal = more complex rhythms
            
            # Harmony mappings
            "chord_complexity": lambda v, a: 0.2 + 0.6 * a,  # Higher arousal = more complex harmony
            "dissonance_level": lambda v, a: max(0, 0.8 * (1 - v) + 0.2 * a),  # Low valence = more dissonance
            "chord_progression_speed": lambda v, a: 0.5 + 0.8 * a,  # Higher arousal = faster changes
            "modulation_frequency": lambda v, a: 0.1 * a,  # Arousal drives modulation
            
            # Texture mappings
            "voice_density": lambda v, a: 2 + 4 * a,  # Higher arousal = more voices
            "layer_complexity": lambda v, a: 0.2 + 0.6 * a,  # Higher arousal = more layering
            "articulation": lambda v, a: v,  # Higher valence = smoother articulation
            "rhythmic_diversity": lambda v, a: 0.3 + 0.4 * a,  # Higher arousal = more diversity
            
            # Dynamics mappings
            "overall_volume": lambda v, a: 0.4 + 0.4 * a,  # Higher arousal = louder
            "dynamic_range": lambda v, a: 0.3 + 0.5 * a,  # Higher arousal = more dynamic
            "crescendo_rate": lambda v, a: 0.2 + 0.6 * a,  # Higher arousal = faster changes
            "accent_strength": lambda v, a: 0.2 + 0.6 * a,  # Higher arousal = stronger accents
            
            # Timbre mappings
            "brightness": lambda v, a: 0.3 + 0.5 * v,  # Higher valence = brighter
            "warmth": lambda v, a: v,  # Higher valence = warmer
            "roughness": lambda v, a: max(0, 0.6 * (1 - v)),  # Lower valence = rougher
            "reverb_amount": lambda v, a: 0.2 + 0.3 * (1 - a),  # Lower arousal = more reverb
            "filter_cutoff": lambda v, a: 0.4 + 0.4 * v,  # Higher valence = more open
            
            # Form mappings
            "section_length": lambda v, a: 16 + 32 * (1 - a),  # Lower arousal = longer sections
            "transition_smoothness": lambda v, a: v,  # Higher valence = smoother transitions
            "repetition_factor": lambda v, a: 0.8 - 0.3 * a,  # Lower arousal = more repetition
            "development_rate": lambda v, a: 0.2 + 0.6 * a  # Higher arousal = faster development
        }
    
    def map_emotion_to_parameters(self, valence: float, arousal: float) -> Dict[str, float]:
        """Convert valence-arousal values to music parameters"""
        # Ensure inputs are in valid range
        valence = np.clip(valence, 0, 1)
        arousal = np.clip(arousal, 0, 1)
        
        # Apply mappings
        params = {}
        for param_name, mapping_func in self.mappings.items():
            params[param_name] = mapping_func(valence, arousal)

        
        # Clip to valid ranges
        return self.param_space.clip_parameters(params)
    
    def get_parameter_sensitivity(self, param_name: str) -> Tuple[float, float]:
        """Get parameter sensitivity to valence and arousal changes"""
        # Compute numerical gradients
        base_v, base_a = 0.5, 0.5
        epsilon = 0.01
        
        base_val = self.mappings[param_name](base_v, base_a)
        
        # Valence sensitivity
        val_plus = self.mappings[param_name](base_v + epsilon, base_a)
        val_sensitivity = (val_plus - base_val) / epsilon
        
        # Arousal sensitivity
        arousal_plus = self.mappings[param_name](base_v, base_a + epsilon)
        arousal_sensitivity = (arousal_plus - base_val) / epsilon
        
        return val_sensitivity, arousal_sensitivity

