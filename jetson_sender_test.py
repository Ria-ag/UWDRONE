import socket

UDP_IP = "192.168.1.100"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

message = "Hello from Jetson!"

sock.sendto(message.encode(), (UDP_IP, UDP_PORT))

print("Message sent!")
sock.close()

