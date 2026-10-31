
from pymavlink import mavutil
import pygame
import sys

# Connect to PX4 SITL
master = mavutil.mavlink_connection(
    "udpin:0.0.0.0:14540"
)

print("Waiting for PX4...")
if master.wait_heartbeat(timeout=15) is None:
    print("Connection timed out")
    sys.exit(1)

print("Connected to PX4 SITL!")

pygame.init()
screen = pygame.display.set_mode((500, 300))
pygame.display.set_caption("PX4 Gazebo Keyboard Test")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 28)

running = True

try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()

        # Desired movement directions
        forward = int(keys[pygame.K_w]) - int(keys[pygame.K_s])
        right = int(keys[pygame.K_d]) - int(keys[pygame.K_a])
        up = int(keys[pygame.K_UP]) - int(keys[pygame.K_DOWN])
        yaw = int(keys[pygame.K_e]) - int(keys[pygame.K_q])

        screen.fill((25, 30, 40))

        lines = [
            "PX4 Gazebo Keyboard Test",
            f"Forward / Backward: {forward}",
            f"Left / Right: {right}",
            f"Up / Down: {up}",
            f"Yaw: {yaw}",
            "WASD, Arrow Keys, Q/E",
        ]

        for i, line in enumerate(lines):
            text = font.render(line, True, (255, 255, 255))
            screen.blit(text, (25, 20 + i * 40))

        pygame.display.flip()
        clock.tick(30)

finally:
    pygame.quit()
    master.close()
