"""
Object Detector Module
======================
Uses YOLOv8 (Ultralytics) to detect objects in video frames and maps
COCO class IDs to three project categories: person, vehicle, and animal.

Usage:
    detector = ObjectDetector(config['object_detection'])
    detections = detector.detect(frame, frame_index=0)
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from ultralytics import YOLO
from src import Detection


# ============================================================
# Object Detector Class
# ============================================================

class ObjectDetector:
    """
    Detects objects in video frames using the YOLOv8 model and maps
    each detection to one of three project categories: person, vehicle,
    or animal.

    Attributes:
        model: Loaded YOLOv8 model instance.
        confidence_threshold: Minimum confidence to keep a detection.
        iou_threshold: IoU threshold for Non-Maximum Suppression.
        device: Inference device ('cpu' or 'cuda').
        _class_id_to_category: Mapping from COCO class ID to category name.
    """

    def __init__(self, config: Dict) -> None:
        """
        Initialise the object detector with YOLO model and config.

        Args:
            config: Dictionary from settings.yaml under 'object_detection'.
                    Expected keys: model_path, confidence_threshold,
                    iou_threshold, device, person_classes, vehicle_classes,
                    animal_classes.
        """
        # --- Load model ---
        model_path: str = config.get("model_path", "yolov8n.pt")
        self.model: YOLO = YOLO(model_path)
        self.config = config

        # --- Detection thresholds ---
        self.confidence_threshold: float = config.get("confidence_threshold", 0.5)
        self.iou_threshold: float = config.get("iou_threshold", 0.45)
        self.device: str = config.get("device", "cpu")

        # --- Build COCO class-ID → category lookup table ---
        # Each entry maps a COCO integer ID to the category string
        self._class_id_to_category: Dict[int, str] = {}

        for class_id in config.get("person_classes", [0]):
            self._class_id_to_category[class_id] = "person"

        for class_id in config.get("vehicle_classes", [2, 3, 5, 7]):
            self._class_id_to_category[class_id] = "vehicle"

        for class_id in config.get("animal_classes", []):
            self._class_id_to_category[class_id] = "animal"

        print(f"[ObjectDetector] Model '{model_path}' loaded on '{self.device}'")
        print(f"[ObjectDetector] Tracking {len(self._class_id_to_category)} COCO classes")

    # ============================================================
    # Public Methods
    # ============================================================

   
    def detect(self, frame: np.ndarray, frame_index: int = 0) -> List[Detection]:
        """
        Detect and track objects using YOLOv8 and ByteTrack.
        """
        detections: List[Detection] = []

        tracking_enabled = self.config.get("tracking_enabled", True)

        if tracking_enabled:
            results = self.model.track(
                frame,
                persist=True,
                tracker=self.config.get("tracker", "bytetrack.yaml"),
                conf=self.config.get("tracking_inference_confidence", 0.1),
                iou=self.iou_threshold,
                device=self.device,
                classes=list(self._class_id_to_category.keys()),
                verbose=False,
            )
        else:
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )

        if not results:
            return detections

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        for i in range(len(boxes)):
            class_id = int(boxes.cls[i].item())
            confidence = float(boxes.conf[i].item())

            mapped = self._map_class(class_id)

            if mapped is None:
                continue

            if confidence < self.confidence_threshold:
                continue

            class_name, class_label = mapped

            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            bbox = (int(x1), int(y1), int(x2), int(y2))

            track_id = None

            if tracking_enabled and boxes.id is not None:
                track_id = int(boxes.id[i].item())

            detection = Detection(
                bbox=bbox,
                class_name=class_name,
                class_label=class_label,
                confidence=round(confidence, 4),
                frame_index=frame_index,
                track_id=track_id,
            )

            detections.append(detection)

        return detections


    # ============================================================
    # Private / Helper Methods
    # ============================================================

    def _map_class(self, class_id: int) -> Optional[Tuple[str, str]]:
        """
        Map a COCO class ID to the project category and specific label.

        Args:
            class_id: Integer COCO class ID (e.g. 0 for person, 2 for car).

        Returns:
            Tuple of (class_name, class_label) where:
                - class_name is the category: 'person', 'vehicle', or 'animal'
                - class_label is the COCO name: 'car', 'dog', 'person', etc.
            Returns None if the class_id is not in our tracked categories.
        """
        # Check if this class ID belongs to any of our categories
        category: Optional[str] = self._class_id_to_category.get(class_id)
        if category is None:
            return None

        # Get the specific COCO label from the model's class names dict
        # model.names is a dict like {0: 'person', 1: 'bicycle', ...}
        coco_label: str = self.model.names.get(class_id, "unknown")

        return (category, coco_label)
