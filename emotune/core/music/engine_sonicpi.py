from typing import Dict, Any
from .engine_base import MusicEngineBase
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer
from emotune.utils.logging import get_logger
import time

class SonicPiEngine(MusicEngineBase):
    def __init__(self, osc_host='127.0.0.1', osc_port=4560, receive_port=4561):
        self.client = SimpleUDPClient(osc_host, osc_port)
        self.receive_port = receive_port
        self.connected = False
        self.logger = get_logger()
        
        # Test connection on init
        self._verify_connection()

    def _verify_connection(self):
        """Verify Sonic Pi is running and listening for OSC messages"""
        dispatcher = Dispatcher()
        dispatcher.map("/ping-reply", self._handle_ping_reply)
        
        try:
            server = ThreadingOSCUDPServer(('127.0.0.1', self.receive_port), dispatcher)
            server.timeout = 2.0  # Add 2 second timeout
            self.client.send_message("/ping", [])
            
            # Wait briefly for response with timeout
            server.handle_request()
            if not self.connected:
                self.logger.warning("Sonic Pi connection verification failed - no response to ping")
        except Exception as e:
            self.logger.error(f"OSC connection error: {str(e)}")
        finally:
            # Ensure server is closed
            try:
                server.server_close()
            except:
                pass

    def _handle_ping_reply(self, *args):
        self.connected = True
        self.logger.info("Verified Sonic Pi OSC connection")

    def update_parameters(self, params: Dict[str, float]):
        """Update all mapped parameters with error handling"""
        param_map = {
            'tempoBpm': '/tempo',
            'tempoVariation': '/tempo_variation',
            'rhythmComplexity': '/rhythm_complexity',
            'chordComplexity': '/chord_complexity',
            'dissonanceLevel': '/dissonance_level',
            'chordProgressionSpeed': '/progression_speed',
            'modulationFrequency': '/modulation_freq',
            'voiceDensity': '/voice_density',
            'layerComplexity': '/layer_complexity',
            'articulation': '/articulation',
            'rhythmicDiversity': '/rhythm_diversity',
            'overallVolume': '/overall_volume',
            'dynamicRange': '/dynamic_range',
            'crescendoRate': '/crescendo_rate',
            'accentStrength': '/accent_strength',
            'brightness': '/brightness',
            'warmth': '/warmth',
            'roughness': '/roughness',
            'reverbAmount': '/reverb_amount',
            'filterCutoff': '/filter_cutoff',
            'sectionLength': '/section_length',
            'transitionSmoothness': '/transition_smooth',
            'repetitionFactor': '/repetition_factor',
            'developmentRate': '/development_rate'
        }

        for key, value in params.items():
            osc_path = param_map.get(key)
            if osc_path:
                try:
                    self.client.send_message(osc_path, [float(value)])
                    self.logger.debug(f"Sent OSC: {osc_path}={value}")
                except Exception as e:
                    self.logger.error(f"Failed to send {key}={value}: {str(e)}")
            else:
                self.logger.warning(f"No OSC mapping for parameter: {key}")

    def play(self, music_struct: Dict[str, Any]):
        """Update parameters and verify changes were sent"""
        self.logger.info(f"Playing music structure: {music_struct}")
        self.update_parameters(music_struct)
        
        # Send a trigger cue to ensure changes take effect
        try:
            self.client.send_message('/trigger_update', [1])
            self.logger.debug("Sent update trigger cue")
        except Exception as e:
            self.logger.error(f"Failed to send trigger: {str(e)}")

    def stop(self):
        """Stop all Sonic Pi jobs with error handling"""
        try:
            self.client.send_message('/stop-all-jobs', [])
            self.logger.info("Sent stop-all-jobs cue")
        except Exception as e:
            self.logger.error(f"Failed to send stop command: {str(e)}")
