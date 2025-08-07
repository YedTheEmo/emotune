import sys
import traceback

try:
    from pyo import Server, Sine
except ImportError as e:
    print("[ERROR] Could not import pyo. Is it installed? Run: pip install pyo")
    sys.exit(1)

print("[INFO] Starting Pyo audio output test...")

try:
    # Boot the server
    s = Server().boot()

    # --- DEBUGGING: Inspecting the Server object ---
    print("\n[DEBUG] Inspecting the Pyo Server object...")
    print(f"[DEBUG] Type of server object: {type(s)}")
    print(f"[DEBUG] Attributes and methods available: {dir(s)}\n")
    # --- END DEBUGGING ---

    # Safely get server info without crashing
    backend = getattr(s, 'getAudio', lambda: "N/A")()
    device = getattr(s, 'getOutputDeviceName', lambda: "N/A")()
    sr = getattr(s, 'getSr', lambda: "N/A")()
    chnls = getattr(s, 'getNchnls', lambda: "N/A")()

    print(f"[INFO] Pyo server booted. Backend: {backend} | Output Device: {device} | Sample Rate: {sr} | Channels: {chnls}")

    s.start()
    print("[INFO] Server started. Playing a 440 Hz sine wave...")
    a = Sine(freq=440, mul=0.2).out()
    print("[INFO] If you hear a tone, the basic audio output is working.")
    print("[INFO] The Pyo GUI will open. Close the GUI window to stop the test.")
    s.gui(locals())  # This will open the Pyo GUI and keep the sound playing until closed
except Exception as e:
    print(f"[ERROR] Exception occurred: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n[INFO] Test complete.") 