"""
object_detector.py
──────────────────
CHIN-BOT Submodule 5 — Object Detection Engine

Runs YOLOv8 inference on camera frames and publishes
detection results as structured DetectionResult objects.

Dependencies:
    pip install ultralytics opencv-python numpy

Author : Dr C.S. Chin — Newcastle University Singapore
Version: 1.0.0
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("opencv-python not found. Install with: pip install opencv-python")

try:
    import numpy as np
    NP_AVAILABLE = True
except ImportError:
    NP_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("ultralytics not found. Install with: pip install ultralytics")

logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────
DEFAULT_MODEL      = "yolov8n.pt"   # nano — fastest; swap to yolov8s/m/l for accuracy
DEFAULT_CONFIDENCE = 0.50           # minimum detection confidence
DEFAULT_IOU        = 0.45           # IoU threshold for NMS
TARGET_FPS         = 10             # inference rate cap to save CPU
CAMERA_INDEX       = 0              # default USB/built-in camera
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Detection:
    """Single detected object."""
    cls:        str
    confidence: float
    bbox_norm:  tuple[float, float, float, float]  # (x, y, w, h) normalised 0-1
    bbox_px:    tuple[int, int, int, int]           # (x1, y1, x2, y2) pixel coords
    center:     tuple[float, float]                 # (cx, cy) normalised

    def to_dict(self) -> dict:
        return {
            "cls":        self.cls,
            "confidence": round(self.confidence, 3),
            "bbox_norm":  [round(v, 3) for v in self.bbox_norm],
            "center":     [round(v, 3) for v in self.center],
        }


@dataclass
class DetectionFrame:
    """All detections for a single camera frame."""
    detections: list[Detection] = field(default_factory=list)
    timestamp:  float = field(default_factory=time.time)
    frame_id:   int   = 0
    latency_ms: float = 0.0

    @property
    def classes(self) -> list[str]:
        return [d.cls for d in self.detections]

    @property
    def count(self) -> int:
        return len(self.detections)

    def get(self, cls: str) -> list[Detection]:
        return [d for d in self.detections if d.cls.upper() == cls.upper()]

    def to_context_string(self) -> str:
        """Produce a natural-language summary for the LLM system prompt."""
        if not self.detections:
            return "No objects detected."
        lines = [f"  - {d.cls} at ({d.center[0]:.0%}, {d.center[1]:.0%}) — {d.confidence:.0%} confidence"
                 for d in self.detections]
        return f"{self.count} object(s) detected:\n" + "\n".join(lines)


class ObjectDetector:
    """
    Wraps YOLOv8 inference over a live camera feed.

    Parameters
    ----------
    model_path   : path to .pt weights file (downloads automatically if not found)
    confidence   : minimum confidence threshold
    camera_index : OpenCV camera index or RTSP/HTTP URL
    on_frame     : callback(DetectionFrame) called after each inference

    Usage
    -----
        detector = ObjectDetector(on_frame=lambda f: print(f.to_context_string()))
        detector.start()   # blocking
        # or non-blocking:
        import threading
        t = threading.Thread(target=detector.start, daemon=True)
        t.start()
        # later…
        frame = detector.latest_frame
        detector.stop()
    """

    def __init__(
        self,
        model_path:   str   = DEFAULT_MODEL,
        confidence:   float = DEFAULT_CONFIDENCE,
        iou:          float = DEFAULT_IOU,
        camera_index: int | str = CAMERA_INDEX,
        on_frame:     Callable[[DetectionFrame], None] = None,
    ):
        self.confidence    = confidence
        self.iou           = iou
        self.camera_index  = camera_index
        self.on_frame      = on_frame
        self.latest_frame: Optional[DetectionFrame] = None
        self._running      = False
        self._frame_id     = 0

        if YOLO_AVAILABLE:
            logger.info("Loading YOLO model: %s", model_path)
            self._model = YOLO(model_path)
            logger.info("Model loaded. Classes: %d", len(self._model.names))
        else:
            self._model = None
            logger.error("YOLO not available — using mock detections.")

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """Blocking inference loop. Call stop() to exit."""
        if not CV2_AVAILABLE:
            logger.error("OpenCV not available — cannot open camera.")
            return

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            logger.error("Cannot open camera index %s.", self.camera_index)
            return

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

        self._running = True
        min_interval  = 1.0 / TARGET_FPS
        logger.info("Object detection started on camera %s.", self.camera_index)

        while self._running:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                logger.warning("Camera read failed — retrying…")
                time.sleep(0.1)
                continue

            det_frame = self._infer(frame)
            self.latest_frame = det_frame
            self._frame_id += 1

            if self.on_frame:
                self.on_frame(det_frame)

            elapsed = time.time() - t0
            sleep_t = max(0.0, min_interval - elapsed)
            time.sleep(sleep_t)

        cap.release()
        logger.info("Object detection stopped.")

    def stop(self):
        self._running = False

    def infer_image(self, image_path: str) -> DetectionFrame:
        """Run inference on a saved image file."""
        if not CV2_AVAILABLE:
            return DetectionFrame()
        frame = cv2.imread(image_path)
        return self._infer(frame) if frame is not None else DetectionFrame()

    # ── Inference ─────────────────────────────────────────────────────────────

    def _infer(self, frame) -> DetectionFrame:
        if self._model is None:
            return self._mock_frame()

        t0 = time.time()
        h, w = frame.shape[:2]

        results = self._model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf  = float(box.conf[0])
            cls_i = int(box.cls[0])
            cls_n = self._model.names[cls_i].upper()

            bw, bh = x2 - x1, y2 - y1
            detections.append(Detection(
                cls        = cls_n,
                confidence = conf,
                bbox_norm  = (x1/w, y1/h, bw/w, bh/h),
                bbox_px    = (x1, y1, x2, y2),
                center     = ((x1+x2)/(2*w), (y1+y2)/(2*h)),
            ))

        return DetectionFrame(
            detections = detections,
            frame_id   = self._frame_id,
            latency_ms = (time.time() - t0) * 1000,
        )

    def _mock_frame(self) -> DetectionFrame:
        """Return a realistic mock frame when YOLO is unavailable."""
        import random
        mock_objects = [
            ("PERSON", 0.94, (0.10, 0.08, 0.18, 0.70)),
            ("TABLE",  0.91, (0.40, 0.50, 0.35, 0.30)),
            ("LAPTOP", 0.89, (0.42, 0.38, 0.16, 0.12)),
            ("CHAIR",  0.87, (0.55, 0.45, 0.20, 0.40)),
        ]
        detections = []
        for cls, conf, (x, y, bw, bh) in mock_objects:
            if random.random() > 0.1:  # occasionally drop one for realism
                detections.append(Detection(
                    cls=cls, confidence=conf + random.uniform(-0.03, 0.03),
                    bbox_norm=(x, y, bw, bh), bbox_px=(0,0,0,0),
                    center=(x + bw/2, y + bh/2),
                ))
        return DetectionFrame(detections=detections, frame_id=self._frame_id, latency_ms=12.0)


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    def on_frame(f: DetectionFrame):
        print(f"\rFrame {f.frame_id:04d} | {f.count} objects | {f.latency_ms:.1f}ms : "
              + ", ".join(f"{d.cls}({d.confidence:.0%})" for d in f.detections),
              end="", flush=True)

    detector = ObjectDetector(
        model_path="yolov8n.pt",
        confidence=0.5,
        on_frame=on_frame,
    )
    print("Running object detection. Press Ctrl-C to stop.")
    try:
        detector.start()
    except KeyboardInterrupt:
        detector.stop()
        print("\nStopped.")
