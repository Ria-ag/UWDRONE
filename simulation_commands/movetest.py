import asyncio
from grpc.aio import AioRpcError
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw

async def run():
    # 1. Connect to the PX4 Gazebo Simulation
    drone = System()
    print("Waiting for drone to connect...")
    await drone.connect(system_address="udpin://0.0.0.0:14540")
    
    # 2. Verify connection
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"Drone connected successfully")
            break

    # 3. Wait for valid position and home estimates before arming
    print("Waiting for global position and home estimates...")

    async def wait_for_health():
        async for health in drone.telemetry.health():
            print(
                f"Global: {health.is_global_position_ok}, "
                f"Local: {health.is_local_position_ok}, "
                f"Home: {health.is_home_position_ok}"
            )

            if health.is_local_position_ok:
                print("Global position and home estimates OK!")
                break

    try:
        await asyncio.wait_for(wait_for_health(), timeout=60)
    except asyncio.TimeoutError as error:
        raise RuntimeError(
            "Timed out waiting for global position and home estimates; "
            "check PX4's estimator and simulator GPS/home initialization."
        ) from error
    
    # 4. Arm and send the initial setpoint before starting Offboard
    print("-- Arming")
    await drone.action.arm()

    print("-- Setting initial setpoint")
    # NED coordinates are relative to the local origin; negative Down is above it.
    try:
        await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, 0.0, 0.0))
    except AioRpcError as error:
        print(f"Setting the initial offboard setpoint failed: {error.details()}")
        print("-- Disarming")
        await drone.action.disarm()
        raise

    print("-- Starting offboard mode")
    try:
        await drone.offboard.start()
    except OffboardError as error:
        print(f"Starting offboard mode failed with error code: {error._result.result}")
        print("-- Disarming")
        await drone.action.disarm()
        return

    # 5. Fly 5m North and East, 5m above the local origin
    print("-- Flying to [North: 5m, East: 5m, Down: -5m]")
    await drone.offboard.set_position_ned(PositionNedYaw(5.0, 5.0, -5.0, 90.0))
    await asyncio.sleep(8)

    # 6. Stop Offboard and Return to Launch / Land
    print("-- Stopping offboard mode")
    try:
        await drone.offboard.stop()
    except OffboardError as error:
        print(f"Stopping offboard mode failed with error code: {error._result.result}")

    print("-- Landing")
    await drone.action.land()

if __name__ == "__main__":
    # Run the async loop
    asyncio.run(run())
