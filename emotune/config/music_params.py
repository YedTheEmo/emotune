# config/music_params.py

from typing import Dict
from core.music.parameter_space import ParameterRange, MusicParam


MUSIC_PARAMETERS: Dict[str, ParameterRange] = {
    "tempo": ParameterRange(
        min_val=60.0,
        max_val=180.0,
        default=120.0,
        description="Beats per minute (BPM)"
    ),
    "density": ParameterRange(
        min_val=0.0,
        max_val=1.0,
        default=0.5,
        description="Note density or rhythmic activity"
    ),
    "brightness": ParameterRange(
        min_val=0.0,
        max_val=1.0,
        default=0.5,
        description="Timbre brightness, e.g., high-pass filtering"
    ),
    "volume": ParameterRange(
        min_val=0.0,
        max_val=1.0,
        default=0.7,
        description="Overall loudness (dynamics)"
    ),
    "reverb": ParameterRange(
        min_val=0.0,
        max_val=1.0,
        default=0.3,
        description="Amount of reverb (spatialization)"
    ),
    "harmonicity": ParameterRange(
        min_val=0.0,
        max_val=1.0,
        default=0.5,
        description="Consonance/dissonance ratio in harmony"
    ),
    "texture": ParameterRange(
        min_val=0.0,
        max_val=1.0,
        default=0.5,
        description="Polyphonic texture complexity"
    ),
    "attack_time": ParameterRange(
        min_val=0.01,
        max_val=1.0,
        default=0.1,
        description="Envelope attack time for notes"
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

