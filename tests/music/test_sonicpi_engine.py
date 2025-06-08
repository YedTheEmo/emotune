from pythonosc.udp_client import SimpleUDPClient
client = SimpleUDPClient("127.0.0.1", 4560)
client.send_message("/trigger/prophet", [70, 100, 8])