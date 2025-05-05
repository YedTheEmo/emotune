from pythonosc import udp_client
import random
import time

def send_test_osc():
    client = udp_client.SimpleUDPClient("127.0.0.1", 57110)  # Ensure using correct port
    params = {
        'bpm': 120,
        'scale': "major",
        'root': random.randint(0, 11),
        'synth': "pluck",
        'harmony': [0, 2, 4, 5, 7],
        'rhythm': [0.5, 0.25, 0.25, 0.5, 0.25, 0.25]
    }

    # Log the parameters we're sending to ensure correct data
    print(f"Sending OSC messages: {params}")
    try:
        client.send_message("/bpm", params['bpm'])
        client.send_message("/scale", params['scale'])
        client.send_message("/root", params['root'])
        client.send_message("/synth", params['synth'])
        client.send_message("/harmony", params['harmony'])
        client.send_message("/rhythm", params['rhythm'])
        
        print("OSC messages sent successfully!")

    except Exception as e:
        print(f"Error sending OSC messages: {e}")

if __name__ == "__main__":
    while True:
        send_test_osc()  # Send OSC messages in a loop for debugging
        time.sleep(3)  # Sleep for 3 seconds between messages

