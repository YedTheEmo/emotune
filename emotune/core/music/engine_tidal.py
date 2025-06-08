from typing import Dict, Any
from .engine_base import MusicEngineBase
from pythonosc.udp_client import SimpleUDPClient
from emotune.utils.logging import get_logger
from .patterns import generate_tidal_pattern

class TidalEngine(MusicEngineBase):
    def __init__(self, osc_host='127.0.0.1', osc_port=6010):
        self.client = SimpleUDPClient(osc_host, osc_port)

    def update_parameters(self, params: Dict[str, float]):
        # Example: send control parameters as OSC messages (customize as needed)
        for key, value in params.items():
            self.client.send_message(f"/ctrl/{key}", value)

    def play(self, music_struct: Dict[str, Any]):
        # Always generate a pattern if not present
        pattern = music_struct.get('pattern')
        if not pattern:
            # Try to generate from parameters
            params = music_struct.get('parameters', music_struct)
            pattern = generate_tidal_pattern(params)
        if pattern:
            self.client.send_message('/d1', pattern)

    def stop(self):
        # Example: mute all patterns
        self.client.send_message('/mute', 1)
