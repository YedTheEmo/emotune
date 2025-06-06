from typing import Dict, Any
from .engine_base import MusicEngineBase
from pythonosc.udp_client import SimpleUDPClient
from emotune.utils.logging import get_logger

class SonicPiEngine(MusicEngineBase):
    def __init__(self, osc_host='127.0.0.1', osc_port=4557):
        self.client = SimpleUDPClient(osc_host, osc_port)

    def update_parameters(self, params: Dict[str, float]):
        # Sonic Pi can receive OSC for live parameter control (customize as needed)
        for key, value in params.items():
            self.client.send_message(f"/emotune/{key}", value)

    def play(self, music_struct: Dict[str, Any]):
        # Send code as a string to Sonic Pi's /run-code endpoint
        code = music_struct.get('code', '')
        if code:
            self.client.send_message('/run-code', code)

    def stop(self):
        # Send a stop command (customize as needed)
        self.client.send_message('/emotune/stop', 1)
