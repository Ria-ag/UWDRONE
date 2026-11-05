import socket
import json

PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", PORT))

print(f"Listening on UDP port {PORT}...")

while True:
    data, addr = sock.recvfrom(4096)

    try:
        message = json.loads(data.decode("utf-8"))

        assert message["type"] == "test"
        assert message["text"] == "Hello from Jetson!"
        assert message["value"] == 42
        assert isinstance(message["timestamp"], (int, float))

        print(f"Received from {addr}: {message}")
        print("SUCCESS: Message received and parsed correctly!")

    except (UnicodeDecodeError, json.JSONDecodeError,
            KeyError, AssertionError, TypeError) as e:
        print(f"FAILED: Invalid message: {e}")
