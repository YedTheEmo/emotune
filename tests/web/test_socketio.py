import eventlet
eventlet.monkey_patch()

from flask import Flask
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')

@app.route('/')
def index():
    return "Hello, SocketIO with Eventlet!"

if __name__ == "__main__":
    socketio.run(app, debug=True)

