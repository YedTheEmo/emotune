from typing import Dict, Any
from .engine_base import MusicEngineBase
import mido
from emotune.utils.logging import get_logger

class MidiEngine(MusicEngineBase):
    def __init__(self, midi_port_name: str = None):
        self.logger = get_logger()
        self.port = None
        
        try:
            # Try to get available MIDI outputs first
            available_outputs = mido.get_output_names()
            if not available_outputs:
                self.logger.warning("No MIDI output devices available")
                return
            
            if midi_port_name:
                if midi_port_name in available_outputs:
                    self.port = mido.open_output(midi_port_name)
                else:
                    self.logger.warning(f"MIDI port '{midi_port_name}' not found")
                    return
            else:
                # Use first available output
                self.port = mido.open_output(available_outputs[0])
                self.logger.info(f"Using MIDI output: {available_outputs[0]}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize MIDI: {e}")
            self.port = None

    def update_parameters(self, params: Dict[str, float]):
        if not self.port:
            return
        # Example: send control changes (customize as needed)
        for cc, value in params.items():
            try:
                cc_num = int(cc)
                midi_value = int(value * 127)
                msg = mido.Message('control_change', control=cc_num, value=midi_value)
                self.port.send(msg)
            except Exception:
                pass

    def play(self, music_struct: Dict[str, Any]):
        if not self.port:
            return
        # Example: play a sequence of notes
        notes = music_struct.get('notes', [])
        for note in notes:
            msg = mido.Message('note_on', note=note['pitch'], velocity=note.get('velocity', 64))
            self.port.send(msg)
            # Note: add timing/duration handling as needed

    def stop(self):
        if not self.port:
            return
        # Example: send all notes off
        for i in range(16):
            self.port.send(mido.Message('control_change', control=123, value=0, channel=i))
