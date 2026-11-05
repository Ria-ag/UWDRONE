import socket
import json
import time

LAPTOP_IP = "192.168.1.100"  # Replace with laptop IP
PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

message = {
    "type": "test",
    "text": "Hello from Jetson!",
    "value": 42,
    "timestamp": time.time()
}

data = json.dumps(message).encode("utf-8")

sock.sendto(data, (LAPTOP_IP, PORT))
print("Sent:", message)

sock.close()
