
import socket
import json
import time

LAPTOPS = {
    "Laptop 1": ("10.18.73.239", 5005),
    # "Laptop 2": ("192.168.1.102", 5005)
}

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(2)

try:
    for sequence in range(1, 6):
        print(f"\nSending message {sequence}")

        # Send to both devices
        for name, address in LAPTOPS.items():
            message = {
                "message": "Hello from Jetson!",
                "sequence": sequence,
                "target": name
            }

            data = json.dumps(message).encode("utf-8")
            sock.sendto(data, address)

            print(f"Sent to {name}: {address}")

        # Verify acknowledgments
        acknowledged = set()
        deadline = time.monotonic() + 2

        while time.monotonic() < deadline:
            sock.settimeout(max(0.001, deadline - time.monotonic()))

            try:
                data, address = sock.recvfrom(4096)
                response = json.loads(data.decode("utf-8"))

                for name, expected_address in LAPTOPS.items():
                    if (
                        address == expected_address
                        and response.get("status") == "received"
                        and response.get("sequence") == sequence
                    ):
                        acknowledged.add(name)
                        print(f"ACK received from {name}")

            except socket.timeout:
                break
            except (UnicodeDecodeError, ValueError):
                print("Invalid acknowledgment received")

        for name in LAPTOPS:
            if name not in acknowledged:
                print(f"No ACK from {name}")

        time.sleep(1)

finally:
    sock.close()
