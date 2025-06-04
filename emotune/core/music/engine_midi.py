from typing import Dict, Any
from .engine_base import MusicEngineBase
import mido

class MidiEngine(MusicEngineBase):
    def __init__(self, midi_port_name: str = None):
        if midi_port_name:
            self.port = mido.open_output(midi_port_name)
        else:
            self.port = mido.open_output()

    def update_parameters(self, params: Dict[str, float]):
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
        # Example: play a sequence of notes
        notes = music_struct.get('notes', [])
        for note in notes:
            msg = mido.Message('note_on', note=note['pitch'], velocity=note.get('velocity', 64))
            self.port.send(msg)
            # Note: add timing/duration handling as needed

    def stop(self):
        # Example: send all notes off
        for i in range(16):
            self.port.send(mido.Message('control_change', control=123, value=0, channel=i))
