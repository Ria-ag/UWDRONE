"""ArUco helpers that work on both old and new OpenCV.

OpenCV >= 4.7 moved ArUco into the main build with a new API
(cv2.aruco.ArucoDetector, generateImageMarker). Older builds (e.g. the
OpenCV 4.5.x that ships with JetPack 5) only have the legacy functions
(cv2.aruco.detectMarkers, DetectorParameters_create, drawMarker).
Everything here picks whichever API exists.
"""
import cv2
import numpy as np

DEFAULT_DICT = "DICT_4X4_50"


def _require_aruco():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            f"This OpenCV ({cv2.__version__}) has no cv2.aruco module. "
            "Install OpenCV >= 4.7 or the contrib build (opencv-contrib-python), "
            "keeping GStreamer support on the Jetson."
        )
    return cv2.aruco


def get_dictionary(name=DEFAULT_DICT):
    aruco = _require_aruco()
    dict_id = getattr(aruco, name)
    if hasattr(aruco, "getPredefinedDictionary"):
        return aruco.getPredefinedDictionary(dict_id)
    return aruco.Dictionary_get(dict_id)


def make_params():
    aruco = _require_aruco()
    if hasattr(aruco, "DetectorParameters_create"):  # OpenCV < 4.7
        params = aruco.DetectorParameters_create()
    else:
        params = aruco.DetectorParameters()
    # Sub-pixel corners -> steadier centers / size estimates
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    # 4x4 dictionaries false-trigger on textures (pavement, grass, gravel).
    # Ignore tiny candidates: perimeter must be >= 4% of the image's larger side.
    params.minMarkerPerimeterRate = 0.04
    return params


class ArucoDetector:
    """detect(gray_or_bgr) -> list of dicts: id, corners (4x2), cx, cy, side_px."""

    def __init__(self, dict_name=DEFAULT_DICT, allowed_ids=None, min_side_px=30):
        # min_side_px=30 rejected a texture false positive (22 px) in testing while
        # still detecting real 30 px markers. Passing allowed_ids (e.g. the IDs you
        # actually printed) is the strongest false-positive filter.
        aruco = _require_aruco()
        self.dictionary = get_dictionary(dict_name)
        self.params = make_params()
        self.allowed_ids = set(allowed_ids) if allowed_ids is not None else None
        self.min_side_px = min_side_px
        if hasattr(aruco, "ArucoDetector"):
            self._detector = aruco.ArucoDetector(self.dictionary, self.params)
            self._detect = self._detector.detectMarkers
        else:
            self._detect = lambda img: aruco.detectMarkers(img, self.dictionary, parameters=self.params)

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        corners, ids, _ = self._detect(gray)
        markers = []
        if ids is None:
            return markers
        for c, i in zip(corners, ids.flatten()):
            marker_id = int(i)
            if self.allowed_ids is not None and marker_id not in self.allowed_ids:
                continue
            pts = c.reshape(4, 2)
            side = float(np.mean([np.linalg.norm(pts[k] - pts[(k + 1) % 4]) for k in range(4)]))
            if side < self.min_side_px:
                continue
            markers.append({
                "id": marker_id,
                "corners": pts,
                "cx": float(pts[:, 0].mean()),
                "cy": float(pts[:, 1].mean()),
                "side_px": side,
            })
        return markers


def draw_markers(frame, markers, color=(255, 0, 255)):
    """Thick magenta outline + large white label box so IDs are readable on the
    video stream even at a distance (magenta is rare in real scenes, unlike blue)."""
    h, w = frame.shape[:2]
    scale = max(0.8, w / 1280)  # keep labels readable on bigger frames
    thick = max(3, int(round(3 * scale)))
    for m in markers:
        pts = m["corners"].astype(int)
        cv2.polylines(frame, [pts], True, color, thick)
        cv2.circle(frame, (int(m["cx"]), int(m["cy"])), thick + 2, color, -1)  # center point
        text = f"ARUCO {m['id']}"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        # label just above the marker's top edge, clamped inside the frame
        x = int(np.clip(pts[:, 0].min(), 0, max(0, w - tw - 12)))
        y = int(np.clip(pts[:, 1].min() - 10, th + 12, h - base - 6))
        cv2.rectangle(frame, (x, y - th - 10), (x + tw + 12, y + base + 2), (255, 255, 255), -1)
        cv2.rectangle(frame, (x, y - th - 10), (x + tw + 12, y + base + 2), color, 2)
        cv2.putText(frame, text, (x + 6, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)
    return frame


def generate_marker(marker_id, size_px=300, dict_name=DEFAULT_DICT):
    aruco = _require_aruco()
    d = get_dictionary(dict_name)
    if hasattr(aruco, "generateImageMarker"):
        return aruco.generateImageMarker(d, marker_id, size_px)
    return aruco.drawMarker(d, marker_id, size_px)
