
import socket
import json

PORT = 15002

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", PORT))

print("Vision module started")

try:
    while True:
        packet, address = sock.recvfrom(65535)
        data = json.loads(packet.decode("utf-8"))

        if data.get("mavpackettype") == "ATTITUDE":
            print(
                f"Attitude received: "
                f"roll={data['roll']:.3f}, "
                f"pitch={data['pitch']:.3f}"
            )

except KeyboardInterrupt:
    print("\nVision stopped")

finally:
    sock.close()
