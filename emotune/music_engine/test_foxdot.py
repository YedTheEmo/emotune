import subprocess
from time import sleep
from FoxDot import *

# Function to start FoxDot server programmatically
def start_foxdot():
    # This will run the FoxDot server as a subprocess
    subprocess.Popen(["foxdot", "--no-ui"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Wait for the server to initialize (give it a few seconds)
sleep(3)

# Now that the server is running, we can define a FoxDot musical pattern
def play_pattern():
    # This is a simple FoxDot pattern (You can replace this with your own code)
    d1 >> pluck([0, 2, 4, 5], dur=0.5)

    # Let the pattern play for 10 seconds and then stop
    Clock.schedule(lambda: d1.stop(), 10)

# Start the FoxDot server

# Play the pattern
play_pattern()

# Start the FoxDot clock to run the code
Clock.start()

# Allow the music to play for 10 seconds (duration of the pattern) before exiting
sleep(12)  # Give it a little extra time after the music stops

# Optionally stop the clock if you need to
Clock.clear()

