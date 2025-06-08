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
class ParameterRange:
    min_val: float
    max_val: float
    default: float
    description: str

class MusicParameterSpace:
    def __init__(self):
        self.parameters = {
            # Tempo parameters
            "tempo_bpm": ParameterRange(60, 180, 120, "Beats per minute"),
            "tempo_variation": ParameterRange(0.0, 0.3, 0.1, "Tempo fluctuation"),
            "rhythm_complexity": ParameterRange(0.0, 1.0, 0.5, "Rhythmic pattern complexity"),
            
            # Harmony parameters
            "chord_complexity": ParameterRange(0.0, 1.0, 0.5, "Harmonic complexity"),
            "dissonance_level": ParameterRange(0.0, 1.0, 0.2, "Amount of dissonance"),
            "chord_progression_speed": ParameterRange(0.1, 2.0, 1.0, "Chord change rate"),
            "modulation_frequency": ParameterRange(0.0, 1.0, 0.1, "Key change frequency"),
            
            # Texture parameters
            "voice_density": ParameterRange(1.0, 8.0, 3.0, "Number of simultaneous voices"),
            "layer_complexity": ParameterRange(0.0, 1.0, 0.4, "Textural layering"),
            "articulation": ParameterRange(0.0, 1.0, 0.5, "Note articulation style"),
            "rhythmic_diversity": ParameterRange(0.0, 1.0, 0.5, "Rhythmic variation between voices"),
            
            # Dynamics parameters
            "overall_volume": ParameterRange(0.3, 1.0, 0.7, "Master volume"),
            "dynamic_range": ParameterRange(0.0, 1.0, 0.6, "Volume variation range"),
            "crescendo_rate": ParameterRange(0.0, 1.0, 0.3, "Volume change speed"),
            "accent_strength": ParameterRange(0.0, 1.0, 0.4, "Accent emphasis"),
            
            # Timbre parameters
            "brightness": ParameterRange(0.0, 1.0, 0.5, "Spectral brightness"),
            "warmth": ParameterRange(0.0, 1.0, 0.6, "Timbral warmth"),
            "roughness": ParameterRange(0.0, 1.0, 0.2, "Timbral roughness"),
            "reverb_amount": ParameterRange(0.0, 1.0, 0.3, "Reverb level"),
            "filter_cutoff": ParameterRange(0.0, 1.0, 0.7, "Low-pass filter cutoff"),
            
            # Form parameters
            "section_length": ParameterRange(8.0, 64.0, 32.0, "Section duration in beats"),
            "transition_smoothness": ParameterRange(0.0, 1.0, 0.7, "Section transition smoothness"),
            "repetition_factor": ParameterRange(0.0, 1.0, 0.6, "Musical repetition amount"),
            "development_rate": ParameterRange(0.0, 1.0, 0.4, "Musical development speed")
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
    
    def clip_parameters(self, params: Dict[str, float]) -> Dict[str, float]:
        """Ensure all parameters are within valid ranges and log any out-of-bounds values."""
        clipped = {}
        logger = get_logger()
        for name, value in params.items():
            if name in self.parameters:
                param = self.parameters[name]
                clipped_value = np.clip(value, param.min_val, param.max_val)
                if value != clipped_value:
                    logger.warning(f"[MusicParameterSpace] Parameter '{name}' value {value} out of range, clipped to {clipped_value}")
                clipped[name] = clipped_value
            else:
                clipped[name] = value
        return clipped

