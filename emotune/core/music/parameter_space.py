import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
from enum import Enum
from emotune.utils.logging import get_logger

class MusicParam(Enum):
    TEMPO = "tempo"
    HARMONY = "harmony"
    TEXTURE = "texture"
    DYNAMICS = "dynamics"
    TIMBRE = "timbre"
    FORM = "form"

@dataclass
class ParameterSpec:
    """Enhanced parameter specification with therapeutic validation"""
    min_val: float
    max_val: float
    default: float
    therapeutic_min: float = None  # Minimum therapeutically safe value
    therapeutic_max: float = None  # Maximum therapeutically safe value
    clinical_notes: str = ""       # Clinical guidance for parameter
    
    def __post_init__(self):
        """Validate parameter specification and set therapeutic bounds"""
        if self.therapeutic_min is None:
            self.therapeutic_min = self.min_val
        if self.therapeutic_max is None:
            self.therapeutic_max = self.max_val
            
        # Ensure therapeutic bounds are within absolute bounds
        self.therapeutic_min = max(self.therapeutic_min, self.min_val)
        self.therapeutic_max = min(self.therapeutic_max, self.max_val)
        
        # Ensure default is within therapeutic range
        self.default = np.clip(self.default, self.therapeutic_min, self.therapeutic_max)

class MusicParameterSpace:
    """Enhanced parameter space with therapeutic validation and clinical safety"""
    
    def __init__(self):
        # CLINICAL UPDATE: Evidence-based parameter ranges with therapeutic constraints
        # References: Music therapy literature, psychoacoustic research, clinical practice guidelines
        
        self.parameters = {
            # === TEMPO PARAMETERS (Critical for arousal regulation) ===
            "tempo_bpm": ParameterSpec(
                min_val=40.0, max_val=200.0, default=100.0,
                therapeutic_min=60.0, therapeutic_max=160.0,
                clinical_notes="Therapeutic range avoids bradycardic/tachycardic entrainment effects"
            ),
            "tempo_variation": ParameterSpec(
                min_val=0.0, max_val=0.3, default=0.1,
                therapeutic_min=0.0, therapeutic_max=0.2,
                clinical_notes="Excessive variation can cause anxiety in sensitive populations"
            ),
            
            # === RHYTHM PARAMETERS (Affect cognitive load and entrainment) ===
            "rhythm_complexity": ParameterSpec(
                min_val=0.0, max_val=1.0, default=0.5,
                therapeutic_min=0.1, therapeutic_max=0.8,
                clinical_notes="Extreme simplicity/complexity can impair therapeutic engagement"
            ),
            "rhythmic_diversity": ParameterSpec(
                min_val=0.0, max_val=1.0, default=0.3,
                therapeutic_min=0.1, therapeutic_max=0.7,
                clinical_notes="Moderate diversity maintains interest without overwhelming"
            ),
            
            # === HARMONY PARAMETERS (Critical for emotional valence) ===
            "chord_complexity": ParameterSpec(
                min_val=0.0, max_val=1.0, default=0.4,
                therapeutic_min=0.1, therapeutic_max=0.8,
                clinical_notes="Complex harmony can enhance positive emotions but may confuse in depression"
            ),
            "dissonance_level": ParameterSpec(
                min_val=0.0, max_val=1.0, default=0.2,
                therapeutic_min=0.0, therapeutic_max=0.7,
                clinical_notes="High dissonance contraindicated for anxiety disorders"
            ),
            "chord_progression_speed": ParameterSpec(
                min_val=0.1, max_val=2.0, default=0.5,
                therapeutic_min=0.2, therapeutic_max=1.5,
                clinical_notes="Rapid changes can increase arousal, slow changes promote stability"
            ),
            "modulation_frequency": ParameterSpec(
                min_val=0.0, max_val=0.5, default=0.1,
                therapeutic_min=0.0, therapeutic_max=0.3,
                clinical_notes="Frequent modulation can destabilize emotional regulation"
            ),
            
            # === TEXTURE PARAMETERS (Affect cognitive processing) ===
            "voice_density": ParameterSpec(
                min_val=1.0, max_val=8.0, default=2.0,
                therapeutic_min=1.0, therapeutic_max=5.0,
                clinical_notes="High voice density can overwhelm processing in ADHD/autism"
            ),
            "layer_complexity": ParameterSpec(
                min_val=0.0, max_val=1.0, default=0.3,
                therapeutic_min=0.1, therapeutic_max=0.8,
                clinical_notes="Layering affects perceived richness and engagement"
            ),
            "articulation": ParameterSpec(
                min_val=0.0, max_val=1.0, default=0.5,
                therapeutic_min=0.2, therapeutic_max=0.9,
                clinical_notes="Smooth articulation promotes relaxation, crisp promotes alertness"
            ),
            
            # === DYNAMICS PARAMETERS (Critical for arousal and safety) ===
            "overall_volume": ParameterSpec(
                min_val=0.0, max_val=1.0, default=0.6,
                therapeutic_min=0.2, therapeutic_max=0.8,
                clinical_notes="Volume limits prevent startle response and hearing damage"
            ),
            "dynamic_range": ParameterSpec(
                min_val=0.0, max_val=1.0, default=0.4,
                therapeutic_min=0.1, therapeutic_max=0.7,
                clinical_notes="Wide dynamics can trigger anxiety, narrow range can seem lifeless"
            ),
            "crescendo_rate": ParameterSpec(
                min_val=0.0, max_val=1.0, default=0.3,
                therapeutic_min=0.1, therapeutic_max=0.8,
                clinical_notes="Sudden changes contraindicated for PTSD/anxiety"
            ),
            "accent_strength": ParameterSpec(
                min_val=0.0, max_val=1.0, default=0.3,
                therapeutic_min=0.0, therapeutic_max=0.7,
                clinical_notes="Strong accents can trigger startle response"
            ),
            
            # === TIMBRE PARAMETERS (Affect emotional perception) ===
            "brightness": ParameterSpec(
                min_val=0.0, max_val=1.0, default=0.5,
                therapeutic_min=0.2, therapeutic_max=0.9,
                clinical_notes="Brightness correlates with perceived energy and hope"
            ),
            "warmth": ParameterSpec(
                min_val=0.0, max_val=1.0, default=0.6,
                therapeutic_min=0.3, therapeutic_max=0.9,
                clinical_notes="Warmth promotes comfort and safety feelings"
            ),
            "roughness": ParameterSpec(
                min_val=0.0, max_val=1.0, default=0.2,
                therapeutic_min=0.0, therapeutic_max=0.6,
                clinical_notes="Roughness can express negative emotions but may increase distress"
            ),
            "reverb_amount": ParameterSpec(
                min_val=0.0, max_val=1.0, default=0.3,
                therapeutic_min=0.0, therapeutic_max=0.7,
                clinical_notes="Reverb creates spaciousness but excessive amounts can feel isolating"
            ),
            "filter_cutoff": ParameterSpec(
                min_val=0.0, max_val=1.0, default=0.6,
                therapeutic_min=0.2, therapeutic_max=0.9,
                clinical_notes="Open sound promotes clarity, filtered sound can feel muffled"
            ),
            
            # === FORM PARAMETERS (Affect cognitive processing and predictability) ===
            "section_length": ParameterSpec(
                min_val=4.0, max_val=64.0, default=16.0,
                therapeutic_min=8.0, therapeutic_max=32.0,
                clinical_notes="Section length affects memory load and predictability"
            ),
            "transition_smoothness": ParameterSpec(
                min_val=0.0, max_val=1.0, default=0.5,
                therapeutic_min=0.2, therapeutic_max=0.9,
                clinical_notes="Smooth transitions reduce anxiety, abrupt ones increase alertness"
            ),
            "repetition_factor": ParameterSpec(
                min_val=0.0, max_val=1.0, default=0.6,
                therapeutic_min=0.3, therapeutic_max=0.9,
                clinical_notes="Repetition provides comfort but excess can cause boredom"
            ),
            "development_rate": ParameterSpec(
                min_val=0.0, max_val=1.0, default=0.4,
                therapeutic_min=0.1, therapeutic_max=0.8,
                clinical_notes="Development rate affects engagement and cognitive challenge"
            ),
            
            # === THERAPEUTIC SAFETY PARAMETERS ===
            "therapeutic_intensity": ParameterSpec(
                min_val=0.0, max_val=1.0, default=0.5,
                therapeutic_min=0.2, therapeutic_max=0.9,
                clinical_notes="Overall therapeutic impact - monitor for over/under-stimulation"
            ),
            "emotional_stability": ParameterSpec(
                min_val=0.0, max_val=1.0, default=0.6,
                therapeutic_min=0.3, therapeutic_max=0.9,
                clinical_notes="Stability prevents emotional dysregulation"
            ),
            "cognitive_load": ParameterSpec(
                min_val=0.0, max_val=1.0, default=0.4,
                therapeutic_min=0.1, therapeutic_max=0.8,
                clinical_notes="Cognitive load must match patient's processing capacity"
            ),
        }
        
        self.parameter_groups = {
            MusicParam.TEMPO: ["tempo_bpm", "tempo_variation", "rhythm_complexity"],
            MusicParam.HARMONY: ["chord_complexity", "dissonance_level", "chord_progression_speed", "modulation_frequency"],
            MusicParam.TEXTURE: ["voice_density", "layer_complexity", "articulation", "rhythmic_diversity"],
            MusicParam.DYNAMICS: ["overall_volume", "dynamic_range", "crescendo_rate", "accent_strength"],
            MusicParam.TIMBRE: ["brightness", "warmth", "roughness", "reverb_amount", "filter_cutoff"],
            MusicParam.FORM: ["section_length", "transition_smoothness", "repetition_factor", "development_rate"]
        }
    
    def get_default_parameters(self) -> Dict[str, float]:
        
        
        return {name: param.default for name, param in self.parameters.items()}

    def normalize_parameter(self, param_name: str, value: float) -> float:
        """Normalize parameter to [0, 1] range"""
        param = self.parameters[param_name]
        return (value - param.min_val) / (param.max_val - param.min_val)
    
    def denormalize_parameter(self, param_name: str, normalized_value: float) -> float:
        """Convert normalized [0, 1] value back to parameter range"""
        param = self.parameters[param_name]
        return param.min_val + normalized_value * (param.max_val - param.min_val)
    
    def clip_parameters(self, params: Dict[str, float], use_therapeutic_bounds: bool = True) -> Dict[str, float]:
        """Enhanced parameter clipping with therapeutic bounds"""
        clipped = {}
        therapeutic_warnings = []
        
        for name, value in params.items():
            if name in self.parameters:
                spec = self.parameters[name]
                
                if use_therapeutic_bounds:
                    # Use therapeutic bounds for clinical safety
                    min_bound = spec.therapeutic_min
                    max_bound = spec.therapeutic_max
                    
                    # Check if value exceeds therapeutic bounds
                    if value < spec.therapeutic_min or value > spec.therapeutic_max:
                        therapeutic_warnings.append(
                            f"{name}={value:.3f} outside therapeutic range "
                            f"[{spec.therapeutic_min:.2f}, {spec.therapeutic_max:.2f}]. "
                            f"Clinical note: {spec.clinical_notes}"
                        )
                else:
                    # Use absolute bounds
                    min_bound = spec.min_val
                    max_bound = spec.max_val
                
                clipped[name] = np.clip(value, min_bound, max_bound)
            else:
                # Unknown parameter - pass through but warn
                clipped[name] = value
                therapeutic_warnings.append(f"Unknown parameter {name} - no therapeutic validation")
        
        # Log therapeutic warnings
        if therapeutic_warnings:
            import warnings
            for warning in therapeutic_warnings[:3]:  # Limit to first 3 warnings
                warnings.warn(f"Therapeutic bounds exceeded: {warning}")
        
        return clipped
    
    def validate_therapeutic_safety(self, params: Dict[str, float]) -> Tuple[bool, list]:
        """Validate parameters for therapeutic safety and return warnings"""
        warnings = []
        is_safe = True
        
        # Critical safety checks
        if params.get('overall_volume', 0.5) > 0.8:
            warnings.append("CRITICAL: Volume too high - risk of hearing damage or startle response")
            is_safe = False
        
        if params.get('dissonance_level', 0.2) > 0.7 and params.get('overall_volume', 0.5) > 0.6:
            warnings.append("CRITICAL: High dissonance + high volume may cause distress")
            is_safe = False
        
        if params.get('tempo_bpm', 100) > 180:
            warnings.append("WARNING: Very high tempo may cause anxiety or agitation")
        
        if params.get('tempo_bpm', 100) < 50:
            warnings.append("WARNING: Very low tempo may induce depressive states")
        
        # Cognitive load assessment
        cognitive_factors = [
            params.get('rhythm_complexity', 0.5),
            params.get('voice_density', 2.0) / 8.0,  # Normalize to 0-1
            params.get('chord_complexity', 0.4),
            params.get('layer_complexity', 0.3)
        ]
        total_cognitive_load = sum(cognitive_factors) / len(cognitive_factors)
        
        if total_cognitive_load > 0.8:
            warnings.append("WARNING: High cognitive load may overwhelm processing capacity")
        
        # Emotional stability assessment
        destabilizing_factors = [
            params.get('tempo_variation', 0.1),
            params.get('dissonance_level', 0.2),
            params.get('modulation_frequency', 0.1),
            1.0 - params.get('transition_smoothness', 0.5)  # Rough transitions destabilize
        ]
        instability_score = sum(destabilizing_factors) / len(destabilizing_factors)
        
        if instability_score > 0.6:
            warnings.append("WARNING: High instability factors may disrupt emotional regulation")
        
        return is_safe, warnings
    
    def get_therapeutic_recommendations(self, valence: float, arousal: float) -> Dict[str, str]:
        """Get therapeutic parameter recommendations based on emotional state"""
        recommendations = {}
        
        # Arousal-based recommendations
        if arousal > 0.8:  # High arousal - need calming
            recommendations.update({
                "tempo_bpm": "Use lower tempo (70-100 BPM) to reduce arousal",
                "dissonance_level": "Minimize dissonance to avoid overstimulation",
                "overall_volume": "Keep volume moderate to prevent startle response",
                "repetition_factor": "Increase repetition for grounding effect"
            })
        elif arousal < 0.3:  # Low arousal - need gentle activation
            recommendations.update({
                "tempo_bpm": "Use moderate tempo (90-120 BPM) to gently increase energy",
                "brightness": "Increase brightness to promote alertness",
                "rhythm_complexity": "Add mild rhythmic interest to engage attention"
            })
        
        # Valence-based recommendations  
        if valence < -0.5:  # Negative valence - need mood support
            recommendations.update({
                "warmth": "Increase warmth for comfort and safety",
                "dissonance_level": "Minimize dissonance to avoid increasing distress",
                "brightness": "Maintain some brightness for hope and energy",
                "chord_complexity": "Use simpler harmonies for emotional clarity"
            })
        elif valence > 0.5:  # Positive valence - can handle more complexity
            recommendations.update({
                "chord_complexity": "Can use richer harmonies to enhance positive emotions",
                "voice_density": "Can support more voices for richness",
                "dynamic_range": "Can use wider dynamics for expressiveness"
            })
        
        return recommendations

