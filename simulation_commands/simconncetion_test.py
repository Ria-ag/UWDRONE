
from pymavlink import mavutil
import sys
import math

# PX4 SITL MAVLink UDP connection
connection_string = "udpin:0.0.0.0:14540"

print(f"Connecting to PX4 SITL on {connection_string}...")

try:
    master = mavutil.mavlink_connection(connection_string)
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
            type="ATTITUDE",
            blocking=True,
            timeout=5
        )

        if msg:
            roll = math.degrees(msg.roll)
            pitch = math.degrees(msg.pitch)
            yaw = math.degrees(msg.yaw)

            print(
                f"\rRoll: {roll:.2f}° | "
                f"Pitch: {pitch:.2f}° | "
                f"Yaw: {yaw:.2f}°   "
            )
            sys.stdout.flush()

except KeyboardInterrupt:
    print("\nExiting...")

finally:
    master.close()
