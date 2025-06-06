from typing import Dict, Any
from emotune.utils.logging import get_logger

class MusicEngineBase:
    def update_parameters(self, params: Dict[str, float]):
        """Update music parameters in real time."""
        raise NotImplementedError

    def play(self, music_struct: Dict[str, Any]):
        """Play a structured music representation."""
        raise NotImplementedError

    def stop(self):
        """Stop playback."""
        raise NotImplementedError
