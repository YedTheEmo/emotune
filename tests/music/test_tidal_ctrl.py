"""
Test sending OSC control messages to TidalCycles to modulate a parameter in a running pattern.
This is the correct way to interact with TidalCycles via OSC (not by sending pattern strings).

Instructions:
1. In your TidalCycles REPL, run:
   d1 $ sound "bd*4" # speed (cF 1 "hello")
2. Run this script. It will send OSC messages to port 6010, address /ctrl, to set the 'hello' control.
3. You should hear the speed of the pattern change.

If you hear the change, the OSC pipeline is working!
"""
import time
from pythonosc.udp_client import SimpleUDPClient

TIDAL_HOST = "127.0.0.1"
TIDAL_CTRL_PORT = 6010
OSC_ADDRESS = "/ctrl"

CONTROL_NAME = "testgain"
CONTROL_VALUES = [0.5, 1.0, 2.0, 0.25]

client = SimpleUDPClient(TIDAL_HOST, TIDAL_CTRL_PORT)

print("[TEST] Sending OSC control messages to TidalCycles...")
for value in CONTROL_VALUES:
    print(f"[TEST] Setting {CONTROL_NAME} to {value}")
    client.send_message(OSC_ADDRESS, [CONTROL_NAME, value])
    time.sleep(2)
print("[TEST] Done. If you heard the pattern speed change, OSC control is working!")
