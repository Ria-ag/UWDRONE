#!/usr/bin/env python3
"""
stereo_camera_node.py
ROS 2 stereo publisher for the Waveshare IMX219-83 (dual IMX219) on a Jetson Orin Nano.

What it fixes:
  * Frame sync   - grabs both sensors in parallel, pairs the newest frames, and stamps
                   BOTH images with the SAME ROS time (ESS drops pairs whose stamps differ).
                   Pairs whose capture times are further apart than max_pair_dt_ms are skipped.
  * Exposure     - both sensors get the same fixed, short exposure and gain range
                   (less motion blur / rolling-shutter smear, identical brightness for matching).
  * Aspect ratio - center-crops to 5:3 so ESS's resize to 960x576 / 480x288 doesn't stretch.
  * Rectify      - in "rect" mode, applies your stereo calibration and publishes rectified
                   images + matching rectified CameraInfo (what ESS expects).

Modes:
  mode:=raw   -> left/image_raw, right/image_raw, left/camera_info, right/camera_info
                 (use this for calibration)
  mode:=rect  -> left/image_rect, right/image_rect, left/camera_info, right/camera_info
                 (rectified; needs left_yaml / right_yaml from camera_calibration)

Example:
  python3 stereo_camera_node.py --ros-args -p mode:=raw
  python3 stereo_camera_node.py --ros-args -p mode:=rect \
      -p left_yaml:=$HOME/stereo_calib/left.yaml -p right_yaml:=$HOME/stereo_calib/right.yaml
"""
import threading
import time

import cv2
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


def gst_pipeline(sensor_id, width, height, fps, exposure_ns, gain_max):
    # Fixed exposure range (min == max) disables auto-exposure drift between the two sensors.
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} "
        f'exposuretimerange="{exposure_ns} {exposure_ns}" '
        f'gainrange="1 {gain_max}" ispdigitalgainrange="1 1" ! '
        f"video/x-raw(memory:NVMM),width={width},height={height},"
        f"framerate={fps}/1,format=NV12 ! "
        f"nvvidconv ! video/x-raw,format=BGRx ! "
        f"videoconvert ! video/x-raw,format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )


class Grabber(threading.Thread):
    """Continuously reads one camera and keeps only the newest frame + its capture time."""

    def __init__(self, pipeline, name):
        super().__init__(daemon=True)
        self.name_ = name
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open {name} camera. Check jetson-io overlay, cable, and that "
                f"OpenCV was built with GStreamer.\nPipeline:\n{pipeline}"
            )
        self.lock = threading.Lock()
        self.frame = None
        self.t = 0.0
        self.seq = 0
        self.running = True

    def run(self):
        while self.running:
            ok, frame = self.cap.read()
            t = time.monotonic()
            if ok:
                with self.lock:
                    self.frame, self.t, self.seq = frame, t, self.seq + 1

    def latest(self):
        with self.lock:
            return self.frame, self.t, self.seq

    def stop(self):
        self.running = False
        self.cap.release()


def load_calib(path):
    with open(path) as f:
        c = yaml.safe_load(f)

    def mat(key, rows, cols):
        return np.array(c[key]["data"], dtype=np.float64).reshape(rows, cols)

    return {
        "width": int(c["image_width"]),
        "height": int(c["image_height"]),
        "K": mat("camera_matrix", 3, 3),
        "D": np.array(c["distortion_coefficients"]["data"], dtype=np.float64),
        "R": mat("rectification_matrix", 3, 3),
        "P": mat("projection_matrix", 3, 4),
    }


def crop_box(w, h, aspect):
    """Center crop (x0, y0, cw, ch) giving width/height == aspect."""
    if w / h > aspect:          # too wide -> crop width
        cw, ch = int(round(h * aspect)), h
    else:                       # too tall -> crop height (IMX219 4:3 lands here)
        cw, ch = w, int(round(w / aspect))
    cw -= cw % 2
    ch -= ch % 2
    return (w - cw) // 2, (h - ch) // 2, cw, ch


class StereoCameraNode(Node):
    def __init__(self):
        super().__init__("imx219_stereo")
        p = self.declare_parameter
        self.mode = p("mode", "raw").value                  # "raw" or "rect"
        left_id = p("left_sensor_id", 0).value
        right_id = p("right_sensor_id", 1).value
        width = p("width", 1640).value                      # 1640x1232@30 = full FOV (binned)
        height = p("height", 1232).value                    # 1280x720@60 = narrower FOV, faster readout
        self.fps = p("fps", 30).value
        exposure_us = p("exposure_us", 3000).value          # 3 ms; raise indoors if too dark
        gain_max = p("gain_max", 8.0).value
        aspect = p("crop_aspect", 5.0 / 3.0).value
        self.max_dt = p("max_pair_dt_ms", 10.0).value / 1000.0
        self.frame_id = p("frame_id", "stereo_left_optical").value

        self.x0, self.y0, self.cw, self.ch = crop_box(width, height, aspect)
        self.get_logger().info(
            f"Capture {width}x{height}@{self.fps} -> crop {self.cw}x{self.ch} "
            f"(offset {self.x0},{self.y0}), mode={self.mode}"
        )

        self.bridge = CvBridge()
        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST)
        suffix = "image_rect" if self.mode == "rect" else "image_raw"
        self.pub_l = self.create_publisher(Image, f"left/{suffix}", qos)
        self.pub_r = self.create_publisher(Image, f"right/{suffix}", qos)
        self.pub_li = self.create_publisher(CameraInfo, "left/camera_info", qos)
        self.pub_ri = self.create_publisher(CameraInfo, "right/camera_info", qos)

        if self.mode == "rect":
            self._setup_rectification(p("left_yaml", "").value, p("right_yaml", "").value)
        else:
            self.info_l = self._blank_info()
            self.info_r = self._blank_info()

        exp_ns = int(exposure_us * 1000)
        self.gl = Grabber(gst_pipeline(left_id, width, height, self.fps, exp_ns, gain_max), "left")
        self.gr = Grabber(gst_pipeline(right_id, width, height, self.fps, exp_ns, gain_max), "right")
        self.gl.start()
        self.gr.start()

        self.last_seq = (0, 0)
        self.stats = {"pub": 0, "skew_drop": 0, "max_skew_ms": 0.0}
        self.create_timer(1.0 / self.fps, self.tick)
        self.create_timer(5.0, self.report)

    # ---------- calibration / rectification ----------
    def _setup_rectification(self, left_yaml, right_yaml):
        if not left_yaml or not right_yaml:
            raise RuntimeError("mode:=rect needs left_yaml and right_yaml parameters")
        cl, cr = load_calib(left_yaml), load_calib(right_yaml)
        for name, c in (("left", cl), ("right", cr)):
            if (c["width"], c["height"]) != (self.cw, self.ch):
                raise RuntimeError(
                    f"{name} calibration is {c['width']}x{c['height']} but cropped images are "
                    f"{self.cw}x{self.ch}. Calibrate in mode:=raw with the SAME capture settings."
                )
        size = (self.cw, self.ch)
        self.map_l = cv2.initUndistortRectifyMap(cl["K"], cl["D"], cl["R"], cl["P"][:, :3], size, cv2.CV_16SC2)
        self.map_r = cv2.initUndistortRectifyMap(cr["K"], cr["D"], cr["R"], cr["P"][:, :3], size, cv2.CV_16SC2)
        self.info_l = self._rect_info(cl["P"])
        self.info_r = self._rect_info(cr["P"])
        fx = cr["P"][0, 0]
        baseline = -cr["P"][0, 3] / fx
        self.get_logger().info(f"Rectified fx={fx:.1f}px at {self.cw}px wide, baseline={baseline*1000:.1f} mm")
        if not 0.05 < baseline < 0.07:
            self.get_logger().warn("Baseline is far from the 60 mm spec - check calibration square size / units")

    def _rect_info(self, P):
        info = CameraInfo()
        info.width, info.height = self.cw, self.ch
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        info.k = P[:, :3].flatten().tolist()
        info.r = np.eye(3).flatten().tolist()
        info.p = P.flatten().tolist()
        info.header.frame_id = self.frame_id
        return info

    def _blank_info(self):
        info = CameraInfo()
        info.width, info.height = self.cw, self.ch
        info.header.frame_id = self.frame_id
        return info

    # ---------- main loop ----------
    def tick(self):
        fl, tl, sl = self.gl.latest()
        fr, tr, sr = self.gr.latest()
        if fl is None or fr is None:
            return
        if sl == self.last_seq[0] or sr == self.last_seq[1]:
            return  # wait for a fresh frame from both sides
        skew = abs(tl - tr)
        self.stats["max_skew_ms"] = max(self.stats["max_skew_ms"], skew * 1000)
        if skew > self.max_dt:
            self.stats["skew_drop"] += 1
            return
        self.last_seq = (sl, sr)

        y0, x0, ch, cw = self.y0, self.x0, self.ch, self.cw
        l = fl[y0:y0 + ch, x0:x0 + cw]
        r = fr[y0:y0 + ch, x0:x0 + cw]
        if self.mode == "rect":
            l = cv2.remap(l, *self.map_l, cv2.INTER_LINEAR)
            r = cv2.remap(r, *self.map_r, cv2.INTER_LINEAR)
        l = cv2.cvtColor(l, cv2.COLOR_BGR2RGB)
        r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)

        stamp = self.get_clock().now().to_msg()  # ONE stamp for both images
        ml = self.bridge.cv2_to_imgmsg(l, encoding="rgb8")
        mr = self.bridge.cv2_to_imgmsg(r, encoding="rgb8")
        for m in (ml, mr, self.info_l, self.info_r):
            m.header.stamp = stamp
            m.header.frame_id = self.frame_id
        self.pub_l.publish(ml)
        self.pub_r.publish(mr)
        self.pub_li.publish(self.info_l)
        self.pub_ri.publish(self.info_r)
        self.stats["pub"] += 1

    def report(self):
        s = self.stats
        self.get_logger().info(
            f"published {s['pub']/5.0:.1f} pairs/s, dropped {s['skew_drop']} for skew, "
            f"max L/R skew {s['max_skew_ms']:.1f} ms"
        )
        self.stats = {"pub": 0, "skew_drop": 0, "max_skew_ms": 0.0}

    def destroy_node(self):
        self.gl.stop()
        self.gr.stop()
        super().destroy_node()


def main():
    rclpy.init()
    node = StereoCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
