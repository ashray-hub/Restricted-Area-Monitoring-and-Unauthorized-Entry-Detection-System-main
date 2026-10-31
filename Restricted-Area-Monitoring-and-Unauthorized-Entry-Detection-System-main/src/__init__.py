"""
Restricted Area Monitoring & Unauthorized Entry Detection System
================================================================

This package contains the core modules for detecting unauthorized entry
in restricted areas from CCTV footage.

Modules:
    - preprocessor: Frame preprocessing (blur, enhance, normalize)
    - motion_detector: Motion detection within ROI using background subtraction
    - roi_manager: Region of Interest polygon management
    - object_detector: YOLOv8 object detection and classification
    - face_detector: Face detection using MediaPipe
    - plate_detector: Number plate detection and OCR
    - animal_classifier: Animal type classification
    - decision_engine: Combines all results and generates verdicts
    - utils: Drawing helpers and utility functions
"""

from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


@dataclass
class Detection:
    """
    Represents a single object detected by the YOLO model.
    
    Attributes:
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        class_name: Detected class category - "person", "vehicle", or "animal"
        class_label: Specific COCO class label (e.g., "car", "dog", "person")
        confidence: Detection confidence score (0.0 to 1.0)
        frame_index: Index of the frame where detection occurred
    """
    bbox: tuple           # (x1, y1, x2, y2)
    class_name: str       # "person", "vehicle", "animal"
    class_label: str      # Specific label like "car", "dog"
    confidence: float     # 0.0 - 1.0
    frame_index: int      # Frame number


@dataclass
class PersonResult:
    """
    Result of person detection with face analysis.
    
    Attributes:
        detection: The original YOLO detection
        face_detected: Whether a face was found in the person crop
        face_bbox: Face bounding box within the original frame
        face_image: Cropped face image as numpy array
    """
    detection: Detection
    face_detected: bool = False
    face_bbox: Optional[tuple] = None       # (x1, y1, x2, y2)
    face_image: Optional[np.ndarray] = None
    is_authorized: bool = False
    identity: Optional[str] = None
    face_distance: Optional[float] = None



@dataclass
class VehicleResult:
    """
    Result of vehicle detection with number plate analysis.
    
    Attributes:
        detection: The original YOLO detection
        plate_detected: Whether a number plate was found
        plate_bbox: Plate bounding box within the original frame
        plate_text: OCR-extracted text from the plate
        plate_image: Cropped plate image as numpy array
    """
    detection: Detection
    plate_detected: bool = False
    plate_bbox: Optional[tuple] = None      # (x1, y1, x2, y2)
    plate_text: Optional[str] = None
    plate_image: Optional[np.ndarray] = None
    is_authorized: bool = False
    identity: Optional[str] = None
    # True when the vehicle's ground-contact point is inside the restricted polygon.
    # Defaults to True for compatibility with existing callers that only pass ROI vehicles.
    in_restricted_zone: bool = True



@dataclass
class AnimalResult:
    """
    Result of animal detection with classification.
    
    Attributes:
        detection: The original YOLO detection
        animal_type: Specific animal type (e.g., "dog", "cat", "bear")
        category: "domestic" or "wild"
        risk_level: Risk assessment ("low", "medium", "high")
    """
    detection: Detection
    animal_type: str = "unknown"
    category: str = "unknown"       # "domestic" or "wild"
    risk_level: str = "medium"      # "low", "medium", "high"


@dataclass
class FrameResult:
    """
    Combined results for a single frame.
    
    Attributes:
        frame_index: Index of the processed frame
        motion_detected: Whether motion was detected in the ROI
        persons: List of person detection results
        vehicles: List of vehicle detection results
        animals: List of animal detection results
        annotated_frame: Frame with all annotations drawn
    """
    frame_index: int
    motion_detected: bool = False
    persons: List[PersonResult] = field(default_factory=list)
    # Vehicles inside the restricted area; retained for existing dashboard summaries.
    vehicles: List[VehicleResult] = field(default_factory=list)
    animals: List[AnimalResult] = field(default_factory=list)
    annotated_frame: Optional[np.ndarray] = None
    # All tracked vehicle observations from the full frame, including those outside the zone.
    # This separate list allows entry/exit tracking without inflating the inside-zone summary.
    vehicle_observations: List[VehicleResult] = field(default_factory=list)
