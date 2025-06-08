"""
Test script for cue-based Sonic Pi emotion music system.
Sends a sequence of parameter cues to Sonic Pi, which must be running the code from 'sonicpi_emotune_livecode.rb'.
"""
from pythonosc.udp_client import SimpleUDPClient
import time

client = SimpleUDPClient("127.0.0.1", 4560)

# Example: Switch between patterns and modulate parameters
print("Setting pattern to 0 (C major)")
client.send_message("/pattern_select", [0])
time.sleep(4)

print("Setting pattern to 1 (D minor)")
client.send_message("/pattern_select", [1])
time.sleep(4)

print("Setting pattern to 2 (E minor)")
client.send_message("/pattern_select", [2])
time.sleep(4)

print("Modulating tempo and volume")
client.send_message("/tempo", [140])
client.send_message("/overall_volume", [0.9])
time.sleep(4)

print("Modulating brightness and reverb")
client.send_message("/brightness", [1.0])
client.send_message("/reverb_amount", [0.7])
time.sleep(4)

print("Modulating articulation and rhythm complexity")
client.send_message("/articulation", [0.9])
client.send_message("/rhythm_complexity", [0.8])
time.sleep(4)

print("Test complete. Listen for changes in Sonic Pi.")
