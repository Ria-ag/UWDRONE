import asyncio
from mavsdk import System

async def main():
    drone = System()

    # Connect to PX4 SITL
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    print("Waiting for drone...")

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected!")
            break

    # Wait until the drone is ready to arm
    async for health in drone.telemetry.health():
        if health.is_global_position_ok:
            break

    await drone.action.arm()
    await drone.action.takeoff()

    await asyncio.sleep(5)

    await drone.action.land()

asyncio.run(main())