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
        Run YOLOv8 inference on a single frame and return filtered detections.

        Steps:
            1. Run YOLO inference with configured thresholds.
            2. Iterate over each detected box.
            3. Map the COCO class ID to our three categories.
            4. Filter out classes we don't track and low-confidence hits.
            5. Build and return a list of Detection dataclass instances.

        Args:
            frame: BGR image as a numpy array (H, W, 3).
            frame_index: Sequential index of this frame in the video.

        Returns:
            List of Detection objects that passed all filters.
        """
        detections: List[Detection] = []

        # --- Run YOLO inference ---
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,  # Suppress per-frame console output
        )

        # results is a list; take the first (and only) element for a single image
        if not results or len(results) == 0:
            return detections

        result = results[0]

        # If no boxes were detected, return empty list
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        # --- Parse each detection box ---
        boxes = result.boxes
        for i in range(len(boxes)):
            # Extract raw values
            class_id: int = int(boxes.cls[i].item())
            confidence: float = float(boxes.conf[i].item())

            # Map COCO class ID to our category
            mapped = self._map_class(class_id)
            if mapped is None:
                # This class is not in our tracked categories – skip
                continue

            class_name, class_label = mapped

            # Extra confidence filter (in case YOLO was run with a lower conf)
            if confidence < self.confidence_threshold:
                continue

            # Bounding box in (x1, y1, x2, y2) pixel coordinates
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            bbox: tuple = (int(x1), int(y1), int(x2), int(y2))

            # Build Detection dataclass
            detection = Detection(
                bbox=bbox,
                class_name=class_name,
                class_label=class_label,
                confidence=round(confidence, 4),
                frame_index=frame_index,
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
