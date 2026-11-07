#!/usr/bin/env python3
"""
disparity_to_obstacles.py
Converts ESS disparity into metric depth and a LaserScan of forward obstacles that MAVROS
forwards to the Pixhawk as MAVLink OBSTACLE_DISTANCE (topic /mavros/obstacle/send).

What it fixes:
  * Focal-length scaling - ESS outputs disparity at the MODEL resolution (480x288 Light,
    960x576 Full), not your camera resolution. This node rescales fx from the rectified
    CameraInfo to the disparity image width before computing depth = f * B / d.
  * Invalid pixels - pixels ESS marks invalid (-1, below the confidence threshold) are treated
    as UNKNOWN, never as free space.

Calibration check (do this before flying): put a flat, textured board exactly 1.00 m and then
2.00 m in front of the cameras. The node logs the center-patch depth every second, plus the
f/T values inside the DisparityImage so you can see whether ESS already rescaled them.

Example:
  python3 disparity_to_obstacles.py --ros-args -p range_max:=4.0
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, LaserScan
from stereo_msgs.msg import DisparityImage
from cv_bridge import CvBridge


class DisparityToObstacles(Node):
    def __init__(self):
        super().__init__("disparity_to_obstacles")
        p = self.declare_parameter
        self.range_min = p("range_min", 0.3).value
        self.range_max = p("range_max", 4.0).value     # ~3-4 m is the trustworthy range with Light ESS
        self.n_bins = p("n_bins", 32).value
        self.band = p("band_fraction", 0.35).value     # center horizontal band of rows to use
        self.pct = p("percentile", 5.0).value          # robust "nearest" per bin
        self.min_valid = p("min_valid_px", 15).value
        out_topic = p("output_topic", "/mavros/obstacle/send").value
        self.frame_id = p("scan_frame_id", "stereo_front").value

        self.bridge = CvBridge()
        self.fx = self.cx = self.base = self.rect_w = None
        self.create_subscription(CameraInfo, "right/camera_info", self.on_info, 5)
        self.create_subscription(DisparityImage, "disparity", self.on_disp, 5)
        self.pub = self.create_publisher(LaserScan, out_topic, 5)
        self.last_log = 0.0

    def on_info(self, msg):
        P = msg.p
        if P[0] == 0.0:
            return
        self.fx, self.cx, self.rect_w = P[0], P[2], msg.width
        self.base = -P[3] / P[0]   # right camera: P[0,3] = -fx * baseline

    def on_disp(self, msg):
        if self.fx is None or self.base is None or self.base <= 0:
            self.get_logger().warn("Waiting for rectified right/camera_info with a baseline...",
                                   throttle_duration_sec=5.0)
            return
        d = self.bridge.imgmsg_to_cv2(msg.image).astype(np.float32)
        h, w = d.shape[:2]
        scale = w / float(self.rect_w)
        f = self.fx * scale
        cx = self.cx * scale

        valid = d > 0.0
        depth = np.full_like(d, np.nan)
        depth[valid] = f * self.base / d[valid]
        depth[(depth < self.range_min * 0.5)] = np.nan  # reject absurdly close outliers

        # Forward-looking band of rows (skip most of the ground and sky)
        r0 = int(h * (0.5 - self.band / 2))
        r1 = int(h * (0.5 + self.band / 2))
        band = depth[r0:r1, :]

        # Column bins -> angles. LaserScan angles are CCW-positive (left = +), so we
        # emit bins from the rightmost column (angle_min) to the leftmost (angle_max).
        edges = np.linspace(0, w, self.n_bins + 1).astype(int)
        no_obstacle = self.range_max + 1.0   # MAVLink convention: max+1 = nothing seen
        ranges = []
        for i in reversed(range(self.n_bins)):
            vals = band[:, edges[i]:edges[i + 1]]
            vals = vals[np.isfinite(vals)]
            if vals.size < self.min_valid:
                ranges.append(no_obstacle)   # unknown -> report nothing, handle in autopilot config
                continue
            r = float(np.percentile(vals, self.pct))
            ranges.append(r if r <= self.range_max else no_obstacle)

        ang_right = -math.atan2(w - cx, f)
        ang_left = math.atan2(cx, f)
        scan = LaserScan()
        scan.header = msg.header
        scan.header.frame_id = self.frame_id
        scan.angle_min = ang_right
        scan.angle_max = ang_left
        scan.angle_increment = (ang_left - ang_right) / max(self.n_bins - 1, 1)
        scan.range_min = self.range_min
        scan.range_max = self.range_max
        scan.ranges = ranges
        self.pub.publish(scan)

        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_log > 1.0:
            self.last_log = now
            ch, cw = h // 2, w // 2
            patch = depth[ch - 6:ch + 6, cw - 6:cw + 6]
            center = float(np.nanmedian(patch)) if np.isfinite(patch).any() else float("nan")
            valid_pct = 100.0 * valid.mean()
            self.get_logger().info(
                f"center depth {center:.2f} m | valid {valid_pct:.0f}% | disp {w}x{h}, "
                f"f_used={f:.1f}px B={self.base*1000:.1f}mm | ESS msg f={msg.f:.1f} T={msg.t:.4f}"
            )


def main():
    rclpy.init()
    node = DisparityToObstacles()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
