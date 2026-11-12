
from pymavlink import mavutil
import sys
import math
import socket
import json
import time

PX4_CONNECTION = "udpin:0.0.0.0:14540"
CLIENT_PORTS = [15001, 15002]

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print("Connecting to PX4 SITL...")
master = mavutil.mavlink_connection(PX4_CONNECTION)

try:
    master = mavutil.mavlink_connection(PX4_CONNECTION)
except Exception as e:
    print(f"Error connecting: {e}")
    sys.exit(1)

try:
    print("Waiting for heartbeat...")
    master.wait_heartbeat(timeout=15)

    print(
        f"Heartbeat received! "
        f"System ID: {master.target_system}, "
        f"Component ID: {master.target_component}"
    )

    print("Listening for Attitude Data (Ctrl+C to stop)...")

    while True:
        msg = master.recv_match(
            type=["LOCAL_POSITION_NED", "ATTITUDE"],
            blocking=True,
            timeout=1
        )

        if msg is None:
            continue

        data = msg.to_dict()
        data["timestamp"] = time.time()

        payload = json.dumps(data).encode("utf-8")
        print(f"Received from PX4: {data['mavpackettype']}", flush=True)

        for port in CLIENT_PORTS:
            sock.sendto(payload, ("127.0.0.1", port))

except KeyboardInterrupt:
    print("\nStopping connection manager")

finally:
    master.close()
    sock.close()
