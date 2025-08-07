# config/music_params.py

from typing import Dict
from emotune.core.music.parameter_space import ParameterSpec, MusicParam


MUSIC_PARAMETERS: Dict[str, ParameterSpec] = {
    "tempo": ParameterSpec(
        min_val=60.0,
        max_val=180.0,
        default=120.0,
        therapeutic_min=70.0,
        therapeutic_max=160.0,
        clinical_notes="Therapeutic range avoids bradycardic/tachycardic entrainment effects"
    ),
    "density": ParameterSpec(
        min_val=0.0,
        max_val=1.0,
        default=0.5,
        therapeutic_min=0.1,
        therapeutic_max=0.8,
        clinical_notes="Note density affects cognitive load and engagement"
    ),
    "brightness": ParameterSpec(
        min_val=0.0,
        max_val=1.0,
        default=0.5,
        therapeutic_min=0.2,
        therapeutic_max=0.9,
        clinical_notes="Brightness correlates with perceived energy and hope"
    ),
    "volume": ParameterSpec(
        min_val=0.0,
        max_val=1.0,
        default=0.7,
        therapeutic_min=0.2,
        therapeutic_max=0.8,
        clinical_notes="Volume limits prevent startle response and hearing damage"
    ),
    "reverb": ParameterSpec(
        min_val=0.0,
        max_val=1.0,
        default=0.3,
        therapeutic_min=0.0,
        therapeutic_max=0.7,
        clinical_notes="Reverb creates spaciousness but excessive amounts can feel isolating"
    ),
    "harmonicity": ParameterSpec(
        min_val=0.0,
        max_val=1.0,
        default=0.5,
        therapeutic_min=0.1,
        therapeutic_max=0.8,
        clinical_notes="Consonance/dissonance ratio affects emotional perception"
    ),
    "texture": ParameterSpec(
        min_val=0.0,
        max_val=1.0,
        default=0.5,
        therapeutic_min=0.1,
        therapeutic_max=0.8,
        clinical_notes="Texture complexity affects cognitive processing"
    ),
    "attack_time": ParameterSpec(
        min_val=0.01,
        max_val=1.0,
        default=0.1,
        therapeutic_min=0.02,
        therapeutic_max=0.8,
        clinical_notes="Attack time affects perceived articulation and energy"
    )
}


# Optionally group them under high-level categories
MUSIC_PARAMETER_GROUPS: Dict[MusicParam, list[str]] = {
    MusicParam.TEMPO: ["tempo"],
    MusicParam.DYNAMICS: ["volume", "attack_time"],
    MusicParam.TIMBRE: ["brightness", "reverb"],
    MusicParam.HARMONY: ["harmonicity"],
    MusicParam.TEXTURE: ["density", "texture"],
    MusicParam.FORM: []  # Optional; left empty for now
}

