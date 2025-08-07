import numpy as np
from typing import Dict, Tuple
from .parameter_space import MusicParameterSpace
from emotune.utils.logging import get_logger

class BaseMappingEngine:
    def __init__(self):
        self.param_space = MusicParameterSpace()
        self._init_mapping_functions()
    
    def _init_mapping_functions(self):
        """Initialize evidence-based mapping functions from valence-arousal to music parameters"""
        # CLINICAL UPDATE: Research-based mappings validated by music therapy literature
        # References: Gomez & Danuser (2007), Krumhansl (1997), Music Therapy Research
        
        self.mappings = {
            # === TEMPO MAPPINGS (Strong Research Evidence) ===
            # Research: Tempo strongly correlates with arousal (r=0.76, Gomez & Danuser 2007)
            "tempo_bpm": lambda v, a: 70 + 80 * a,  # 70-150 BPM therapeutic range
            "tempo_variation": lambda v, a: 0.02 + 0.15 * (1 - v) * a,  # Negative valence + high arousal = more variation
            
            # === RHYTHM MAPPINGS (Evidence-Based) ===
            # Research: Rhythm complexity increases with arousal, moderated by valence
            "rhythm_complexity": lambda v, a: 0.2 + 0.6 * a * (0.5 + 0.5 * v),  # Positive valence enhances rhythm acceptance
            "rhythmic_diversity": lambda v, a: 0.1 + 0.4 * a,  # Simple arousal correlation
            
            # === HARMONY MAPPINGS (Strong Clinical Evidence) ===
            # Research: Major/minor mode strongly affects valence (Krumhansl 1997)
            "chord_complexity": lambda v, a: 0.3 + 0.4 * a + 0.2 * v,  # Both valence and arousal contribute
            # FIXED DISSONANCE: Research shows dissonance increases with negative valence AND high arousal
            "dissonance_level": lambda v, a: np.clip(0.1 + 0.4 * (1 - v) + 0.3 * a * (1 - v), 0, 0.8),  # Interaction effect
            "chord_progression_speed": lambda v, a: 0.3 + 0.6 * a,  # Arousal-driven harmonic rhythm
            "modulation_frequency": lambda v, a: 0.05 + 0.15 * a * (1 - v),  # Modulation for arousal, less for positive valence
            
            # === TEXTURE MAPPINGS (Therapeutic Considerations) ===
            # Research: Voice density affects perceived complexity and arousal
            "voice_density": lambda v, a: 1 + 3 * a * (0.3 + 0.7 * v),  # 1-4 voices, positive valence allows more
            "layer_complexity": lambda v, a: 0.1 + 0.5 * a,  # Simple arousal relationship
            "articulation": lambda v, a: 0.2 + 0.6 * v + 0.2 * (1 - a),  # Smooth for positive valence, calm arousal
            
            # === DYNAMICS MAPPINGS (Psychoacoustic Evidence) ===
            # Research: Loudness correlates with arousal, but therapeutic limits apply
            "overall_volume": lambda v, a: np.clip(0.3 + 0.4 * a + 0.1 * v, 0.2, 0.8),  # Therapeutic volume limits
            "dynamic_range": lambda v, a: 0.2 + 0.4 * a * (0.5 + 0.5 * v),  # Positive valence allows more dynamics
            "crescendo_rate": lambda v, a: 0.1 + 0.5 * a,  # Arousal-driven dynamic changes
            "accent_strength": lambda v, a: 0.1 + 0.4 * a * (0.3 + 0.7 * v),  # Positive valence enhances accents
            
            # === TIMBRE MAPPINGS (Spectral Research) ===
            # Research: Brightness (spectral centroid) correlates with both valence and arousal
            "brightness": lambda v, a: 0.2 + 0.4 * v + 0.3 * a,  # FIXED: Both dimensions contribute
            "warmth": lambda v, a: 0.1 + 0.7 * v + 0.2 * (1 - a),  # Warm = positive valence, low arousal
            "roughness": lambda v, a: np.clip(0.1 + 0.5 * (1 - v) + 0.2 * a, 0, 0.6),  # Negative valence + arousal
            # Research: Reverb creates spaciousness, therapeutic for low arousal states
            "reverb_amount": lambda v, a: 0.1 + 0.4 * (1 - a) + 0.2 * v,  # Low arousal + positive valence
            "filter_cutoff": lambda v, a: 0.3 + 0.4 * v + 0.2 * a,  # Open sound for positive states
            
            # === FORM MAPPINGS (Cognitive Load Theory) ===
            # Research: Section length affects cognitive processing and therapeutic outcomes
            "section_length": lambda v, a: 8 + 24 * (1 - a) * (0.5 + 0.5 * v),  # 8-32 beats, longer for calm positive states
            "transition_smoothness": lambda v, a: 0.2 + 0.6 * v + 0.2 * (1 - a),  # Smooth for positive, calm states
            "repetition_factor": lambda v, a: 0.4 + 0.4 * (1 - a) + 0.2 * v,  # Repetition for calm, positive states
            "development_rate": lambda v, a: 0.1 + 0.6 * a * (0.5 + 0.5 * v),  # Fast development for high arousal, positive valence
            
            # === THERAPEUTIC SAFETY MAPPINGS ===
            # New parameters for clinical safety and therapeutic effectiveness
            "therapeutic_intensity": lambda v, a: np.clip(0.3 + 0.4 * a + 0.3 * v, 0.2, 0.9),  # Overall therapeutic impact
            "emotional_stability": lambda v, a: 0.3 + 0.4 * v + 0.3 * (1 - a),  # Stability for positive, calm states
            "cognitive_load": lambda v, a: 0.2 + 0.5 * a * (1 - v),  # Higher load for high arousal, negative valence
        }
        
        # Ensure all parameters in parameter space are mapped
        for name in self.param_space.parameters:
            if name not in self.mappings:
                # Default: return the default value with slight valence bias
                default_val = self.param_space.parameters[name].default
                # Add slight therapeutic bias toward positive outcomes
                self.mappings[name] = lambda v, a, d=default_val: d * (0.8 + 0.2 * v)
    
    def map_emotion_to_parameters(self, valence: float, arousal: float) -> Dict[str, float]:
        """Convert valence-arousal values to music parameters with clinical validation."""
        # CLINICAL FIX: Ensure inputs are in therapeutic range
        # Valence: [-1, 1] (negative to positive emotions)
        # Arousal: [0, 1] (calm to excited) - FIXED from previous [-1,1] range
        valence = np.clip(valence, -1.0, 1.0)
        arousal = np.clip(arousal, 0.0, 1.0)

        # Apply evidence-based mappings
        params = {}
        for param_name, mapping_func in self.mappings.items():
            try:
                raw_value = mapping_func(valence, arousal)
                # Clinical validation: ensure no NaN or infinite values
                if np.isnan(raw_value) or np.isinf(raw_value):
                    raw_value = self.param_space.parameters[param_name].default
                params[param_name] = raw_value
            except Exception as e:
                # Fallback to default for any mapping errors
                params[param_name] = self.param_space.parameters[param_name].default
                import warnings
                warnings.warn(f"Mapping error for {param_name}: {e}. Using default.")

        # Clip to therapeutic ranges
        params = self.param_space.clip_parameters(params)

        # Clinical validation: ensure therapeutic appropriateness
        params = self._apply_therapeutic_constraints(params, valence, arousal)

        # Ensure all parameters are present
        for name, param in self.param_space.parameters.items():
            if name not in params:
                params[name] = param.default

        return params
    
    def _apply_therapeutic_constraints(self, params: Dict[str, float], valence: float, arousal: float) -> Dict[str, float]:
        """Apply therapeutic constraints to ensure clinical safety and effectiveness."""
        # Therapeutic constraint 1: Limit extreme dissonance for negative emotional states
        if valence < -0.5 and params.get('dissonance_level', 0) > 0.6:
            params['dissonance_level'] = 0.6  # Prevent overwhelming dissonance
        
        # Therapeutic constraint 2: Ensure minimum warmth for very negative states
        if valence < -0.7:
            params['warmth'] = max(params.get('warmth', 0), 0.3)  # Provide some comfort
        
        # Therapeutic constraint 3: Limit volume for high arousal states (prevent overstimulation)
        if arousal > 0.8:
            params['overall_volume'] = min(params.get('overall_volume', 0.5), 0.7)
        
        # Therapeutic constraint 4: Ensure sufficient repetition for very high arousal (grounding)
        if arousal > 0.9:
            params['repetition_factor'] = max(params.get('repetition_factor', 0.5), 0.6)
        
        # Therapeutic constraint 5: Minimum brightness for very low valence (hope)
        if valence < -0.8:
            params['brightness'] = max(params.get('brightness', 0.3), 0.25)
        
        return params
    
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

