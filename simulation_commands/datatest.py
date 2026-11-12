
from pymavlink import mavutil
import math
import time

CONNECTION = "udpin:0.0.0.0:14540"

class DroneController:
    def __init__(self):
        self.master = mavutil.mavlink_connection(CONNECTION)
        print("Waiting for PX4 SITL...")

        if self.master.wait_heartbeat(timeout=15) is None:
            raise TimeoutError("PX4 connection timed out")

        print("Connected to PX4!")
        self.position = None

    def update(self):
        # Read available telemetry
        while True:
            msg = self.master.recv_match(blocking=False)

            if msg is None:
                break

            if msg.get_type() == "LOCAL_POSITION_NED":
                self.position = (msg.x, msg.y, msg.z)

    def calculate_velocity(self, target, speed=1.0):
        if self.position is None:
            return None

        difference = [
            target[i] - self.position[i]
            for i in range(3)
        ]

        distance = math.sqrt(
            sum(value**2 for value in difference)
        )

        if distance < 0.1:
            return (0.0, 0.0, 0.0)

        return tuple(
            speed * value / distance
            for value in difference
        )

    def close(self):
        self.master.close()


def main():
    drone = DroneController()

    # Example AI-generated waypoint
    target = (5.0, 3.0, -2.0)

    try:
        while True:
            drone.update()

            velocity = drone.calculate_velocity(target)

            if velocity is not None:
                print(
                    f"Position: {drone.position} | "
                    f"Desired velocity: {velocity}"
                )

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        drone.close()


if __name__ == "__main__":
    main()
