"""
🔒 Restricted Area Monitoring & Unauthorized Entry Detection System
================================================================
Pipeline Integration & Sanity Check Test Script

This script initializes all components and processes a few frames of the downloaded
sample video to verify that everything works correctly together.

Author: Computer Vision Project Team
"""

import cv2
import os
import sys
import numpy as np

# Add project root to path (parent of this tests directory)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils import load_config, format_timestamp
from src.preprocessor import FramePreprocessor
from src.motion_detector import MotionDetector
from src.roi_manager import ROIManager
from src.object_detector import ObjectDetector
from src.face_detector import FaceDetector
from src.plate_detector import PlateDetector
from src.animal_classifier import AnimalClassifier
from src.decision_engine import DecisionEngine, Alert
from src import FrameResult

def main():
    print("🚀 Starting integration test for Restricted Area Monitoring System...")
    
    # 1. Paths (resolved dynamically from project root)
    video_path = os.path.join(project_root, "data", "sample.mp4")
    config_path = os.path.join(project_root, "config", "settings.yaml")
    
    if not os.path.exists(video_path):
        print(f"❌ Error: Sample video not found at {video_path}")
        sys.exit(1)
        
    if not os.path.exists(config_path):
        print(f"❌ Error: Config file not found at {config_path}")
        sys.exit(1)
        
    # 2. Load Config
    print("📖 Loading configuration...")
    config = load_config(config_path)
    print("✅ Configuration loaded successfully.")
    
    # 3. Initialize Video Info
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📹 Video Info: {width}x{height} | FPS: {fps} | Total Frames: {total_frames}")
    
    # 4. Define a simple rectangular ROI covering the center region of the video
    # Let's say we monitor the center of the frame (where pedestrians and vehicles pass)
    roi_points = [
        (int(width * 0.1), int(height * 0.1)),
        (int(width * 0.9), int(height * 0.1)),
        (int(width * 0.9), int(height * 0.9)),
        (int(width * 0.1), int(height * 0.9))
    ]
    print(f"📍 Defined ROI polygon with {len(roi_points)} points: {roi_points}")
    
    # 5. Initialize Modules with their respective sub-config sections
    print("🔧 Initializing system modules...")
    preprocessor = FramePreprocessor(config.get("preprocessing", {}))
    motion_detector = MotionDetector(config.get("motion_detection", {}))
    roi_manager_original = ROIManager()
    roi_manager_original.set_polygon(roi_points)
    
    resize_width = preprocessor.resize_width
    resize_height = preprocessor.resize_height
    scale_x_down = resize_width / width
    scale_y_down = resize_height / height
    
    roi_points_preprocessed = [
        (int(x * scale_x_down), int(y * scale_y_down))
        for x, y in roi_points
    ]
    roi_manager_preprocessed = ROIManager()
    roi_manager_preprocessed.set_polygon(roi_points_preprocessed)
    
    roi_mask_preprocessed = roi_manager_preprocessed.generate_mask((resize_height, resize_width))
    
    object_detector = ObjectDetector(config.get("object_detection", {}))
    face_detector = FaceDetector(
        config.get("face_detection", {}),
        config.get("authorization", {}),
        config.get("face_verification", {})
    )
    plate_detector = PlateDetector(config.get("plate_detection", {}))
    animal_classifier = AnimalClassifier(config.get("animal_classification", {}))
    decision_engine = DecisionEngine(config)
    
    print("✅ All modules initialized successfully.")
    
    # 6. Process first 30 frames
    print("🎬 Processing first 30 frames...")
    frames_to_process = min(30, total_frames)
    motion_count = 0
    detection_count = 0
    alert_count = 0
    
    for frame_idx in range(frames_to_process):
        ret, frame = cap.read()
        if not ret:
            print("⚠️ End of video reached early.")
            break
            
        # Preprocess
        processed_frame = preprocessor.preprocess(frame)
        
        # Motion detection (using preprocessed mask)
        motion_detected, motion_mask, contours = motion_detector.detect_motion(processed_frame, roi_mask_preprocessed)
        
        frame_result = FrameResult(
            frame_index=frame_idx,
            motion_detected=motion_detected
        )
        
        if motion_detected:
            motion_count += 1
            # Run YOLO
            detections = object_detector.detect(processed_frame, frame_idx)
            
            # Scale YOLO detections back to original dimensions
            scale_x_up = width / resize_width
            scale_y_up = height / resize_height
            
            scaled_detections = []
            for d in detections:
                x1, y1, x2, y2 = d.bbox
                d.bbox = (
                    int(x1 * scale_x_up),
                    int(y1 * scale_y_up),
                    int(x2 * scale_x_up),
                    int(y2 * scale_y_up)
                )
                scaled_detections.append(d)
            
            # Filter in ROI (original coordinates)
            roi_detections = [d for d in scaled_detections if roi_manager_original.is_inside_roi(d.bbox)]
            
            for detection in roi_detections:
                detection_count += 1
                bbox = detection.bbox
                
                if detection.class_name == "person":
                    person_result = face_detector.detect_face(frame, bbox)
                    person_result.detection = detection
                    frame_result.persons.append(person_result)
                    print(f"  👤 Frame {frame_idx}: Detected person (Conf: {detection.confidence:.2f}). Face detected: {person_result.face_detected}")
                    
                elif detection.class_name == "vehicle":
                    vehicle_result = plate_detector.detect_plate(frame, bbox)
                    vehicle_result.detection = detection
                    frame_result.vehicles.append(vehicle_result)
                    print(f"  🚗 Frame {frame_idx}: Detected {detection.class_label} (Conf: {detection.confidence:.2f}). Plate: {vehicle_result.plate_text}")
                    
                elif detection.class_name == "animal":
                    animal_result = animal_classifier.classify(detection)
                    frame_result.animals.append(animal_result)
                    print(f"  🐾 Frame {frame_idx}: Detected animal {animal_result.animal_type} (Category: {animal_result.category}, Risk: {animal_result.risk_level})")
        
        # Decision engine
        alerts = decision_engine.process_frame_result(frame_result, fps)
        if alerts:
            alert_count += len(alerts)
            for alert in alerts:
                print(f"  🔔 [ALERT] Frame {frame_idx} | Severity: {alert.severity} | Msg: {alert.message}")
                
    cap.release()
    face_detector.release()
    print("\n📊 --- Integration Test Summary ---")
    print(f"Total Frames Processed: {frames_to_process}")
    print(f"Frames with Motion: {motion_count}")
    print(f"Total Detections in ROI: {detection_count}")
    print(f"Total Alerts Triggered: {alert_count}")
    print("🎉 Integration test completed successfully without runtime crashes!")

if __name__ == "__main__":
    main()
