
from pymavlink import mavutil
import pygame
import sys
import math

# Simulation-only connection
CONNECTION = "udpin:0.0.0.0:14540"

master = mavutil.mavlink_connection(CONNECTION)

print("Waiting for PX4 SITL...")
if master.wait_heartbeat(timeout=15) is None:
    print("Connection timed out")
    sys.exit(1)

print("Connected!")

pygame.init()
screen = pygame.display.set_mode((540, 360))
pygame.display.set_caption("PX4 SITL Keyboard Controller")
font = pygame.font.SysFont(None, 26)
clock = pygame.time.Clock()

running = True
attitude = None

try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        keys = pygame.key.get_pressed()

        # Intended body-relative movement
        forward = int(keys[pygame.K_w]) - int(keys[pygame.K_s])
        right = int(keys[pygame.K_d]) - int(keys[pygame.K_a])
        up = int(keys[pygame.K_UP]) - int(keys[pygame.K_DOWN])
        yaw = int(keys[pygame.K_e]) - int(keys[pygame.K_q])

        if keys[pygame.K_SPACE]:
            forward = right = up = yaw = 0

        # Read available telemetry without blocking UI
        while True:
            msg = master.recv_match(blocking=False)
            if msg is None:
                break
            if msg.get_type() == "ATTITUDE":
                attitude = (
                    math.degrees(msg.roll),
                    math.degrees(msg.pitch),
                    math.degrees(msg.yaw)
                )

        # Desired movement vector, not transmitted
        desired = (forward, right, up, yaw)

        screen.fill((25, 30, 40))
        lines = [
            "PX4 SITL Keyboard Input",
            f"Forward: {desired[0]}",
            f"Right: {desired[1]}",
            f"Up: {desired[2]}",
            f"Yaw: {desired[3]}",
            "Space: Stop | Esc: Exit",
        ]

        if attitude:
            lines.append(
                "Attitude: "
                + ", ".join(f"{v:.1f}" for v in attitude)
            )

        for i, line in enumerate(lines):
            rendered = font.render(
                line, True, (255, 255, 255)
            )
            screen.blit(rendered, (20, 20 + i * 38))

        pygame.display.flip()
        clock.tick(30)

finally:
    pygame.quit()
    master.close()
