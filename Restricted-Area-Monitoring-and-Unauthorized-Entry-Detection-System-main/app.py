"""
Restricted Area Monitoring & Unauthorized Entry Detection System
================================================================
Main Streamlit Application

This is the web-based dashboard for the restricted area monitoring system.
Users can:
    1. Upload CCTV video files
    2. Draw restricted area polygons on the first frame
    3. Configure detection parameters
    4. Process the video and view results
    5. Browse detection logs and alerts

Author: Computer Vision Project Team
Course: Image Processing & Computer Vision
"""

import streamlit as st
import cv2
import numpy as np
import os
import sys
import tempfile
import time
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_config, draw_bbox, get_color_for_class, format_timestamp
from src.preprocessor import FramePreprocessor
from src.motion_detector import MotionDetector
from src.roi_manager import ROIManager
from src.object_detector import ObjectDetector
from src.face_detector import FaceDetector
from src.plate_detector import PlateDetector
from src.animal_classifier import AnimalClassifier
from src.decision_engine import DecisionEngine, Alert
from src import FrameResult, VehicleResult


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Restricted Area Monitor",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# Custom CSS Styling
# ============================================================

def inject_custom_css():
    """Inject custom CSS for a modern, premium dark theme dashboard."""
    st.markdown("""
    <style>
        /* ---- Import Google Font ---- */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        /* ---- Global Styles ---- */
        .stApp {
            font-family: 'Inter', sans-serif;
        }
        
        /* ---- Main Header ---- */
        .main-header {
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            padding: 2rem 2.5rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        .main-header h1 {
            color: #ffffff;
            font-size: 2rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: -0.5px;
        }
        .main-header p {
            color: rgba(255, 255, 255, 0.65);
            font-size: 1rem;
            margin: 0.5rem 0 0 0;
            font-weight: 300;
        }
        
        /* ---- Status Badge ---- */
        .status-badge {
            display: inline-block;
            padding: 4px 14px;
            border-radius: 50px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        .status-active { background: rgba(34, 197, 94, 0.15); color: #22c55e; border: 1px solid rgba(34, 197, 94, 0.3); }
        .status-warning { background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }
        .status-danger { background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }
        .status-info { background: rgba(59, 130, 246, 0.15); color: #3b82f6; border: 1px solid rgba(59, 130, 246, 0.3); }
        
        /* ---- Metric Cards ---- */
        .metric-card {
            background: linear-gradient(145deg, rgba(30, 30, 50, 0.9), rgba(20, 20, 40, 0.95));
            backdrop-filter: blur(20px);
            padding: 1.5rem;
            border-radius: 14px;
            border: 1px solid rgba(255, 255, 255, 0.06);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
            text-align: center;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
        }
        .metric-value {
            font-size: 2.5rem;
            font-weight: 800;
            margin: 0.5rem 0;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .metric-label {
            color: rgba(255, 255, 255, 0.5);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            font-weight: 600;
        }
        .metric-icon {
            font-size: 1.8rem;
            margin-bottom: 0.3rem;
        }
        
        /* ---- Alert Cards ---- */
        .alert-card {
            padding: 1rem 1.5rem;
            border-radius: 12px;
            margin-bottom: 0.75rem;
            border-left: 4px solid;
            backdrop-filter: blur(10px);
        }
        .alert-critical {
            background: rgba(220, 38, 38, 0.08);
            border-left-color: #dc2626;
        }
        .alert-high {
            background: rgba(239, 68, 68, 0.08);
            border-left-color: #ef4444;
        }
        .alert-medium {
            background: rgba(245, 158, 11, 0.08);
            border-left-color: #f59e0b;
        }
        .alert-low {
            background: rgba(34, 197, 94, 0.08);
            border-left-color: #22c55e;
        }
        .alert-time {
            color: rgba(255, 255, 255, 0.4);
            font-size: 0.75rem;
            font-weight: 500;
        }
        .alert-message {
            color: rgba(255, 255, 255, 0.85);
            font-size: 0.9rem;
            margin: 4px 0;
        }
        .alert-severity {
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 1px;
        }
        
        /* ---- Step Indicator ---- */
        .step-container {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 1rem 1.5rem;
            background: linear-gradient(145deg, rgba(30, 30, 50, 0.8), rgba(20, 20, 40, 0.9));
            border-radius: 12px;
            margin-bottom: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        .step-number {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.9rem;
            flex-shrink: 0;
        }
        .step-text {
            color: rgba(255, 255, 255, 0.85);
            font-size: 1rem;
            font-weight: 500;
        }
        
        /* ---- Section Divider ---- */
        .section-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
            margin: 2rem 0;
        }
        
        /* ---- Processing Box ---- */
        .processing-info {
            background: rgba(59, 130, 246, 0.08);
            border: 1px solid rgba(59, 130, 246, 0.2);
            border-radius: 12px;
            padding: 1rem 1.5rem;
            margin: 1rem 0;
        }
        
        /* ---- Hide default Streamlit styling where needed ---- */
        .block-container { padding-top: 2rem; }
        
        /* ---- Sidebar ---- */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f0c29, #1a1a2e);
        }
        [data-testid="stSidebar"] .stMarkdown h2 {
            color: rgba(255,255,255,0.9);
            font-size: 1.1rem;
        }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# Helper Functions
# ============================================================

def render_header():
    """Render the main application header."""
    st.markdown("""
    <div class="main-header">
        <h1>🔒 Restricted Area Monitoring System</h1>
        <p>Computer Vision-based Unauthorized Entry Detection — Upload CCTV footage, define restricted zones, and detect intrusions automatically</p>
    </div>
    """, unsafe_allow_html=True)


def render_step(number: int, text: str):
    """Render a step indicator with number and description."""
    st.markdown(f"""
    <div class="step-container">
        <div class="step-number">{number}</div>
        <div class="step-text">{text}</div>
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(icon: str, value, label: str):
    """Render a styled metric card."""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def render_alert_card(alert: Alert):
    """Render a styled alert card."""
    severity_class = f"alert-{alert.severity.lower()}"
    severity_colors = {
        "CRITICAL": "#dc2626", "HIGH": "#ef4444",
        "MEDIUM": "#f59e0b", "LOW": "#22c55e"
    }
    color = severity_colors.get(alert.severity, "#3b82f6")
    st.markdown(f"""
    <div class="alert-card {severity_class}">
        <div class="alert-time">🕐 {alert.timestamp} — Frame #{alert.frame_index}</div>
        <div class="alert-message">{alert.message}</div>
        <div class="alert-severity" style="color: {color};">{alert.severity} SEVERITY</div>
    </div>
    """, unsafe_allow_html=True)


def get_video_info(video_path: str) -> dict:
    """
    Extract video metadata using OpenCV.
    
    Args:
        video_path: Path to the video file
    
    Returns:
        Dictionary with video properties (fps, frame_count, width, height, duration)
    """
    cap = cv2.VideoCapture(video_path)
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
    cap.release()
    return info


def extract_first_frame(video_path: str) -> np.ndarray:
    """
    Extract the first frame from a video file.
    
    Args:
        video_path: Path to the video file
    
    Returns:
        First frame as a numpy array (BGR)
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    return None


def canvas_points_to_polygon(canvas_result) -> list:
    """
    Convert Streamlit drawable canvas result to polygon points.
    
    Args:
        canvas_result: Result from st_canvas component
    
    Returns:
        List of (x, y) tuples representing polygon vertices
    """
    if canvas_result.json_data is None:
        return []
    
    objects = canvas_result.json_data.get("objects", [])
    if not objects:
        return []
    
    points = []
    for obj in objects:
        if obj.get("type") == "path":
            # Freeform drawing - extract path points
            path = obj.get("path", [])
            for cmd in path:
                if len(cmd) >= 3 and cmd[0] in ["M", "L"]:
                    x = int(cmd[1] + obj.get("left", 0))
                    y = int(cmd[2] + obj.get("top", 0))
                    points.append((x, y))
                elif len(cmd) >= 3 and cmd[0] == "Q":
                    # Quadratic bezier - take endpoint
                    x = int(cmd[3] + obj.get("left", 0)) if len(cmd) >= 5 else int(cmd[1] + obj.get("left", 0))
                    y = int(cmd[4] + obj.get("top", 0)) if len(cmd) >= 5 else int(cmd[2] + obj.get("top", 0))
                    points.append((x, y))
        elif obj.get("type") == "rect":
            # Rectangle selection
            left = int(obj.get("left", 0))
            top = int(obj.get("top", 0))
            width = int(obj.get("width", 0) * obj.get("scaleX", 1))
            height = int(obj.get("height", 0) * obj.get("scaleY", 1))
            points = [
                (left, top),
                (left + width, top),
                (left + width, top + height),
                (left, top + height)
            ]
    
    return points


# ============================================================
# Video Processing Pipeline
# ============================================================

def process_video(video_path: str, roi_points: list, config: dict,
                  progress_bar, status_text, preview_placeholder=None) -> tuple:
    """
    Main video processing pipeline.
    
    Processes the video frame by frame:
    1. Preprocess each frame (blur, enhance)
    2. Check for motion within the ROI
    3. If motion detected, run YOLO object detection
    4. Run specialized detectors based on object type
    5. Generate alerts through the decision engine
    
    Args:
        video_path: Path to the input video
        roi_points: List of (x,y) polygon points for restricted area
        config: Configuration dictionary
        progress_bar: Streamlit progress bar widget
        status_text: Streamlit text widget for status updates
    
    Returns:
        Tuple of (decision_engine, annotated_frames, output_video_path)
    """
    # ---- Initialize All Modules ----
    status_text.text("🔧 Initializing detection models...")
    
    # Each module expects its own sub-section of the config
    preprocessor = FramePreprocessor(config.get("preprocessing", {}))
    motion_detector = MotionDetector(config.get("motion_detection", {}))
    roi_manager_original = ROIManager()
    object_detector = ObjectDetector(config.get("object_detection", {}))
    face_detector = FaceDetector(
        config.get("face_detection", {}),
        config.get("authorization", {}),
        config.get("face_verification", {})
    )
    plate_detector = PlateDetector(config.get("plate_detection", {}))
    animal_classifier = AnimalClassifier(config.get("animal_classification", {}))
    decision_engine = DecisionEngine(config)
    
    # Set up ROI in original dimensions
    roi_manager_original.set_polygon(roi_points)
    
    # ---- Open Video ----
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Handle video rotation in dimensions calculation
    rotation_mode = config.get("preprocessing", {}).get("video_rotation", "0°")
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if rotation_mode in ["90° Clockwise", "90° Counter-Clockwise"]:
        width, height = orig_height, orig_width
    else:
        width, height = orig_width, orig_height
    
    # Scale ROI points for preprocessed frame size (for motion detection)
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
    
    # Generate preprocessed ROI mask
    roi_mask_preprocessed = roi_manager_preprocessed.generate_mask((resize_height, resize_width))
    
    # ---- Prepare Output Video ----
    output_dir = config.get("output", {}).get("output_directory", "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "annotated_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Storage for key frames (frames with detections)
    key_frames = []
    frame_index = 0
    frame_skip = max(1, int(config.get("preprocessing", {}).get("frame_skip", 1)))
    
    status_text.text("🎬 Processing video frames...")
    
    # ---- Frame Processing Loop ----
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Rotate frame if requested
        if rotation_mode == "90° Clockwise":
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_mode == "180°":
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation_mode == "90° Counter-Clockwise":
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Update progress
        progress = (frame_index + 1) / total_frames
        progress_bar.progress(progress, text=f"Processing frame {frame_index + 1}/{total_frames}")
        
        # Check if this frame should be skipped
        if frame_index % frame_skip != 0:
            # Create a baseline frame result
            frame_result = FrameResult(
                frame_index=frame_index,
                motion_detected=False
            )
            
            # Create annotated frame (copy of original with ROI overlay)
            annotated = frame.copy()
            annotated = roi_manager_original.draw_roi(annotated, color=(0, 255, 0), alpha=0.2)
            
            # Draw frame info
            timestamp = format_timestamp(frame_index / fps if fps > 0 else 0)
            cv2.putText(annotated, f"Frame: {frame_index} | Time: {timestamp} (Skipped)",
                       (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (200, 200, 200), 1)
            
            # Draw skipped indicator in upper-right corner
            cv2.putText(annotated, "FAST PLAYBACK (SKIPPED)", (width - 240, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            
            # Feed basic frame results to the decision engine for correct time/frame indexes
            decision_engine.process_frame_result(frame_result, fps)
            
            # Write to output video
            out.write(annotated)
            
            # Render realtime preview if placeholder is provided
            if preview_placeholder is not None:
                preview_frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                preview_placeholder.image(
                    preview_frame_rgb,
                    caption="🔍 Real-time Analysis Preview (Processing...)",
                    use_container_width=True
                )
                
            frame_index += 1
            continue
        
        # Step 1: Preprocess the frame (resizes to resize_width x resize_height)
        processed_frame = preprocessor.preprocess(frame)
        
        # Step 2: Detect motion within ROI using the preprocessed mask
        motion_detected, motion_mask, contours = motion_detector.detect_motion(
            processed_frame, roi_mask_preprocessed
        )
        
        # Create annotated frame (copy of original)
        annotated = frame.copy()
        
        # Draw ROI overlay on original frame
        annotated = roi_manager_original.draw_roi(annotated, color=(0, 255, 0), alpha=0.2)
        
        # Initialize frame result
        frame_result = FrameResult(
            frame_index=frame_index,
            motion_detected=motion_detected
        )
        
        # Step 3: Draw motion contours when motion is detected inside the ROI.
        scale_x_up = width / resize_width
        scale_y_up = height / resize_height

        if motion_detected:
            # Scale contours back to original dimensions for drawing.
            scaled_contours = []
            for contour in contours:
                scaled_c = contour.copy().astype(np.float32)
                scaled_c[:, 0, 0] *= scale_x_up
                scaled_c[:, 0, 1] *= scale_y_up
                scaled_contours.append(scaled_c.astype(np.int32))
            cv2.drawContours(annotated, scaled_contours, -1, (0, 255, 255), 1)

        # Important for entry tracking: run YOLO on every non-skipped frame,
        # not only when motion is found inside the restricted polygon. This lets
        # the tracker observe a vehicle approaching from outside the zone.
        detections = object_detector.detect(processed_frame, frame_index)
        scaled_detections = []
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            detection.bbox = (
                int(x1 * scale_x_up),
                int(y1 * scale_y_up),
                int(x2 * scale_x_up),
                int(y2 * scale_y_up),
            )
            scaled_detections.append(detection)

        # Process all vehicle detections for track continuity. Vehicle zone
        # membership is measured at the bottom-centre of its bounding box.
        for detection in scaled_detections:
            bbox = detection.bbox

            if detection.class_name == "vehicle":
                x1, y1, x2, y2 = bbox
                vehicle_point = ((x1 + x2) / 2.0, float(y2))
                in_restricted_zone = roi_manager_original.is_point_inside_roi(vehicle_point)

                # Run plate OCR only for vehicles in the zone on frames where
                # motion was detected; otherwise keep a lightweight observation
                # for tracking without paying the OCR cost on every frame.
                if in_restricted_zone and motion_detected:
                    vehicle_result = plate_detector.detect_plate(frame, bbox)
                    vehicle_result.detection = detection
                    vehicle_result.in_restricted_zone = True

                    # Check plate authorization for UI display.
                    if vehicle_result.plate_detected and vehicle_result.plate_text:
                        import json
                        import re
                        auth_file = config.get("authorization", {}).get(
                            "authorized_plates_file", "config/authorized_plates.json"
                        )
                        auth_path = auth_file
                        if not os.path.isabs(auth_path):
                            auth_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), auth_path)
                        authorized_plates = []
                        if os.path.exists(auth_path):
                            try:
                                with open(auth_path, "r", encoding="utf-8") as f:
                                    plates = json.load(f)
                                    if isinstance(plates, list):
                                        authorized_plates = [str(p).strip().upper() for p in plates]
                            except Exception as e:
                                print(f"[app] Error loading authorized plates: {e}")

                        clean_ocr = re.sub(r"[^A-Z0-9]", "", vehicle_result.plate_text.upper())
                        for auth_plate in authorized_plates:
                            clean_auth = re.sub(r"[^A-Z0-9]", "", auth_plate.upper())
                            if clean_ocr == clean_auth:
                                vehicle_result.is_authorized = True
                                vehicle_result.identity = auth_plate
                                break
                else:
                    vehicle_result = VehicleResult(
                        detection=detection,
                        in_restricted_zone=in_restricted_zone,
                    )

                # All vehicles are passed to the decision engine for tracking.
                frame_result.vehicle_observations.append(vehicle_result)

                if in_restricted_zone:
                    # Keep the existing ROI-only summary/key-frame behavior when
                    # the motion pipeline is active. The separate observation list
                    # still updates the tracker on low-motion frames.
                    if motion_detected:
                        frame_result.vehicles.append(vehicle_result)

                    track_label = (
                        f"ID:{detection.track_id}"
                        if detection.track_id is not None
                        else "ID:?"
                    )

                    if vehicle_result.is_authorized:
                        color = (0, 255, 0)
                        label = (
                            f"{detection.class_label.upper()} {track_label} "
                            f"{vehicle_result.identity} (Authorized)"
                        )
                    else:
                        color = (0, 165, 255)
                        label = (
                            f"{detection.class_label.upper()} "
                            f"{track_label} {detection.confidence:.0%}"
                        )
                        if vehicle_result.plate_detected and vehicle_result.plate_text:
                            label += f" [{vehicle_result.plate_text}]"

                    if vehicle_result.plate_detected and vehicle_result.plate_bbox:
                        pb = vehicle_result.plate_bbox
                        plate_box_color = (0, 255, 0) if vehicle_result.is_authorized else (0, 255, 255)
                        cv2.rectangle(annotated, (pb[0], pb[1]), (pb[2], pb[3]), plate_box_color, 2)
                        plate_lbl = (
                            f"PLATE: {vehicle_result.identity}"
                            if vehicle_result.is_authorized
                            else (vehicle_result.plate_text or "PLATE")
                        )
                        cv2.putText(
                            annotated, plate_lbl, (pb[0], pb[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, plate_box_color, 2
                        )
                    draw_bbox(annotated, bbox, label, color)
                else:
                    # Show outside-zone vehicles in blue so entry into the
                    # polygon can be observed visually, but do not include these
                    # detections in the existing in-zone summary.
                    draw_bbox(
                        annotated, bbox,
                        (
                        f"OUTSIDE ZONE - {detection.class_label.upper()} "
                        f"ID:{detection.track_id if detection.track_id is not None else '?'} "
                        f"{detection.confidence:.0%}"
                    ),
                        (255, 120, 0),
                    )

            elif motion_detected and roi_manager_original.is_inside_roi(bbox):
                if detection.class_name == "person":
                    # Person detections remain restricted to the monitored zone.
                    person_result = face_detector.detect_face(frame, bbox)
                    person_result.detection = detection
                    frame_result.persons.append(person_result)

                    dist_lbl = f" ({person_result.face_distance:.2f})" if person_result.face_distance is not None else ""
                    if person_result.is_authorized:
                        color = (0, 255, 0)
                        label = f"👤 {person_result.identity} (Authorized){dist_lbl}"
                    else:
                        color = (0, 0, 255)
                        label = f"PERSON {detection.confidence:.0%}"
                        if person_result.face_detected:
                            label += f" [FACE{dist_lbl}]"

                    if person_result.face_detected and person_result.face_bbox:
                        fb = person_result.face_bbox
                        face_box_color = (0, 255, 0) if person_result.is_authorized else (255, 0, 255)
                        cv2.rectangle(annotated, (fb[0], fb[1]), (fb[2], fb[3]), face_box_color, 2)
                        face_lbl = (
                            f"{person_result.identity} (Authorized){dist_lbl}"
                            if person_result.is_authorized else f"FACE{dist_lbl}"
                        )
                        cv2.putText(
                            annotated, face_lbl, (fb[0], fb[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_box_color, 2
                        )
                    draw_bbox(annotated, bbox, label, color)

                elif detection.class_name == "animal":
                    animal_result = animal_classifier.classify(detection)
                    frame_result.animals.append(animal_result)
                    color = (0, 255, 0)
                    label = f"{animal_result.animal_type.upper()} ({animal_result.category}) {detection.confidence:.0%}"
                    draw_bbox(annotated, bbox, label, color)

        if motion_detected:
            cv2.putText(
                annotated, "MOTION DETECTED", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )

        # Draw frame info
        timestamp = format_timestamp(frame_index / fps if fps > 0 else 0)
        cv2.putText(annotated, f"Frame: {frame_index} | Time: {timestamp}",
                   (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (200, 200, 200), 1)
        
        # Process through decision engine
        alerts = decision_engine.process_frame_result(frame_result, fps)
        
        # Draw alert indicators on frame
        if alerts:
            for i, alert in enumerate(alerts):
                cv2.putText(annotated, f"ALERT: {alert.severity}",
                           (10, 60 + i * 25), cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, (0, 0, 255), 2)
        
        # Save frame result
        frame_result.annotated_frame = annotated
        
        # Store key frames (frames with detections)
        if frame_result.persons or frame_result.vehicles or frame_result.animals:
            key_frames.append(frame_result)
        
        # Write to output video
        out.write(annotated)
        
        # Render realtime preview if placeholder is provided
        if preview_placeholder is not None:
            # Convert BGR to RGB for Streamlit display
            preview_frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            preview_placeholder.image(
                preview_frame_rgb,
                caption="🔍 Real-time Analysis Preview (Processing...)",
                use_container_width=True
            )
        
        frame_index += 1
    
    # ---- Cleanup ----
    cap.release()
    out.release()
    face_detector.release()
    
    # Clear the preview placeholder
    if preview_placeholder is not None:
        preview_placeholder.empty()
    
    status_text.text("✅ Processing complete!")
    progress_bar.progress(1.0, text="Processing complete!")
    
    return decision_engine, key_frames, output_path


def render_authorized_directory_manager(config):
    """Render the Authorized Directory Manager page to view, add, and remove plates and faces."""
    st.markdown("<div class='main-header'><h1>🔑 Authorized Directory Manager</h1><p>Manage authorized vehicle license plates and personnel profiles (DeepFace database)</p></div>", unsafe_allow_html=True)
    
    auth_config = config.get("authorization", {})
    plates_file = auth_config.get("authorized_plates_file", "config/authorized_plates.json")
    faces_dir = auth_config.get("authorized_faces_dir", "data/authorized_faces")
    
    # Resolve absolute paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    plates_path = os.path.join(base_dir, plates_file)
    faces_path = os.path.join(base_dir, faces_dir)
    
    # Ensure directories/files exist
    os.makedirs(os.path.dirname(plates_path), exist_ok=True)
    os.makedirs(faces_path, exist_ok=True)
    
    if not os.path.exists(plates_path):
        import json
        with open(plates_path, "w") as f:
            json.dump([], f)
            
    # Load plates
    import json
    try:
        with open(plates_path, "r", encoding="utf-8") as f:
            plates = json.load(f)
            if not isinstance(plates, list):
                plates = []
    except Exception:
        plates = []
        
    m_tab1, m_tab2 = st.tabs(["🚗 Authorized License Plates", "👤 Authorized Face Directory"])
    
    with m_tab1:
        st.subheader("Manage Approved Vehicle Plates")
        
        col_add, col_space = st.columns([2, 2])
        with col_add:
            new_plate = st.text_input("Add Approved Plate Number", placeholder="e.g. ABC-1234").strip().upper()
            if st.button("➕ Add Plate", use_container_width=True):
                if new_plate:
                    if new_plate not in plates:
                        plates.append(new_plate)
                        try:
                            with open(plates_path, "w", encoding="utf-8") as f:
                                json.dump(plates, f, indent=2)
                            st.success(f"Plate {new_plate} added successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to save plate: {e}")
                    else:
                        st.warning(f"Plate {new_plate} is already authorized.")
                else:
                    st.error("Please enter a valid plate number.")
                    
        st.write("### Currently Authorized Plates")
        if plates:
            # Display plates in a list with delete buttons
            for plate in sorted(plates):
                col_p_text, col_p_btn, _ = st.columns([2, 1, 3])
                col_p_text.markdown(f"**`🚗 {plate}`**")
                if col_p_btn.button("🗑️ Remove", key=f"del_{plate}"):
                    plates.remove(plate)
                    try:
                        with open(plates_path, "w", encoding="utf-8") as f:
                            json.dump(plates, f, indent=2)
                        st.success(f"Plate {plate} removed successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to remove plate: {e}")
        else:
            st.info("No authorized plate numbers found.")
            
    with m_tab2:
        st.subheader("Manage Approved Personnel Profiles")
        
        col_prof_name, col_prof_img = st.columns(2)
        with col_prof_name:
            person_name = st.text_input("Person's Name", placeholder="e.g. John Doe")
            uploaded_faces = st.file_uploader("Upload Profile Image(s) (DeepFace Reference)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
            
            if st.button("➕ Register Person", use_container_width=True):
                if person_name and uploaded_faces:
                    # Sanitize filename: replace spaces with underscores, lowercase
                    safe_name = person_name.strip().replace(" ", "_").lower()
                    
                    # Count existing files starting with the same name to determine next indexes
                    existing_files = [f for f in os.listdir(faces_path) if f.lower().startswith(safe_name)]
                    
                    success_count = 0
                    for idx, uploaded_face in enumerate(uploaded_faces):
                        file_ext = os.path.splitext(uploaded_face.name)[1].lower()
                        if not file_ext:
                            file_ext = ".jpg"
                        
                        next_index = len(existing_files) + 1
                        target_filename = f"{safe_name}_{next_index}{file_ext}"
                        target_file = os.path.join(faces_path, target_filename)
                        
                        try:
                            with open(target_file, "wb") as f:
                                f.write(uploaded_face.getbuffer())
                            existing_files.append(target_filename)
                            success_count += 1
                        except Exception as e:
                            st.error(f"Failed to save image {uploaded_face.name}: {e}")
                    
                    if success_count > 0:
                        # Clear DeepFace cached representation pkl files
                        pkl_files = [f for f in os.listdir(faces_path) if f.endswith('.pkl')]
                        for pkl in pkl_files:
                            try:
                                os.remove(os.path.join(faces_path, pkl))
                            except Exception:
                                pass
                        st.success(f"Successfully registered {person_name} with {success_count} profile images!")
                        st.rerun()
                else:
                    st.error("Please provide both name and profile image(s).")
                    
        with col_prof_img:
            if uploaded_faces:
                st.markdown("<div style='margin-bottom: 5px; font-weight: bold;'>📷 Image Preview:</div>", unsafe_allow_html=True)
                preview_cols = st.columns(min(len(uploaded_faces), 3))
                for idx, uploaded_face in enumerate(uploaded_faces):
                    col_idx = idx % 3
                    with preview_cols[col_idx]:
                        try:
                            st.image(uploaded_face, use_container_width=True)
                        except Exception:
                            st.warning("⚠️ Preview error")
            else:
                st.markdown(
                    """
                    <div style="border: 2px dashed rgba(255, 255, 255, 0.1); border-radius: 8px; height: 160px; display: flex; align-items: center; justify-content: center; color: rgba(255, 255, 255, 0.3);">
                        No images uploaded
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                    
        st.write("### Currently Authorized Profiles")
        face_files = [f for f in os.listdir(faces_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if face_files:
            cols = st.columns(3)
            for idx, file in enumerate(sorted(face_files)):
                col = cols[idx % 3]
                filepath = os.path.join(faces_path, file)
                
                # Format name and strip index suffixes (e.g. dileepa_1 -> Dileepa)
                import re
                name_w_ext = os.path.splitext(file)[0]
                name_clean = re.sub(r'[\d_-]+$', '', name_w_ext)
                name = name_clean.replace("_", " ").title()
                
                with col:
                    st.markdown(f"**👤 {name}** (`{file}`)")
                    try:
                        st.image(filepath, use_container_width=True)
                    except Exception:
                        st.warning("Could not display image.")
                        
                    if st.button("🗑️ Delete Profile", key=f"del_face_{file}"):
                        try:
                            os.remove(filepath)
                            # Clear DeepFace representation pkl files
                            pkl_files = [f for f in os.listdir(faces_path) if f.endswith('.pkl')]
                            for pkl in pkl_files:
                                try:
                                    os.remove(os.path.join(faces_path, pkl))
                                except Exception:
                                    pass
                            st.success(f"Profile for {name} deleted!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to delete profile: {e}")
        else:
            st.info("No authorized face profiles registered.")


# ============================================================
# Main Application
# ============================================================

def main():
    """Main Streamlit application entry point."""
    
    # Inject custom CSS
    inject_custom_css()
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "config", "settings.yaml")
    config = load_config(config_path)
    
    # ---- Sidebar: Navigation & Settings ----
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        app_page = st.radio(
            "Go to:",
            ["🖥️ Surveillance Monitor", "🔑 Authorized Directory Manager"],
            index=0
        )
        st.markdown("---")
        
        if app_page == "🖥️ Surveillance Monitor":
            st.markdown("## ⚙️ Detection Settings")
            st.markdown("---")
            
            # YOLO confidence threshold
            confidence = st.slider(
                "🎯 Detection Confidence",
                min_value=0.1, max_value=1.0,
                value=config.get("object_detection", {}).get("confidence_threshold", 0.5),
                step=0.05,
                help="Minimum confidence score for YOLO detections"
            )
            config.setdefault("object_detection", {})["confidence_threshold"] = confidence

            # Video Rotation
            video_rotation = st.selectbox(
                "🔄 Video Rotation",
                options=["0°", "90° Clockwise", "180°", "90° Counter-Clockwise"],
                index=["0°", "90° Clockwise", "180°", "90° Counter-Clockwise"].index(
                    config.get("preprocessing", {}).get("video_rotation", "0°")
                ),
                help="Rotate the video frame if the camera orientation is sideways or upside down"
            )
            config.setdefault("preprocessing", {})["video_rotation"] = video_rotation

            # Face verification settings
            st.markdown("---")
            st.markdown("## 👤 Face Recognition Settings")
            
            face_threshold = st.slider(
                "🔑 Face Match Threshold",
                min_value=0.1, max_value=1.0,
                value=config.get("face_verification", {}).get("threshold", 0.65),
                step=0.05,
                help="Cosine distance threshold for face matching (lower = stricter, higher = more lenient)"
            )
            config.setdefault("face_verification", {})["threshold"] = face_threshold
            
            # Hardcode model name and detector backend for a simplified OpenCV + DeepFace pipeline
            face_model = "Facenet"
            detector_backend = "opencv"
            config.setdefault("face_verification", {})["model_name"] = face_model
            config.setdefault("face_verification", {})["detector_backend"] = detector_backend

            # Clear pkl cache if model or backend changes (for migration from old configs)
            if "prev_face_model" not in st.session_state:
                st.session_state.prev_face_model = face_model
            if "prev_detector_backend" not in st.session_state:
                st.session_state.prev_detector_backend = detector_backend
                
            if st.session_state.prev_face_model != face_model or st.session_state.prev_detector_backend != detector_backend:
                st.session_state.prev_face_model = face_model
                st.session_state.prev_detector_backend = detector_backend
                
                # Delete representations .pkl files
                faces_dir = config.get("authorization", {}).get("authorized_faces_dir", "data/authorized_faces")
                base_dir = os.path.dirname(os.path.abspath(__file__))
                faces_path = os.path.join(base_dir, faces_dir)
                if os.path.exists(faces_path):
                    pkl_files = [f for f in os.listdir(faces_path) if f.endswith('.pkl')]
                    for pkl in pkl_files:
                        try:
                            os.remove(os.path.join(faces_path, pkl))
                        except Exception:
                            pass
                    st.toast("⚡ Face database cache cleared because model/backend settings changed.")
            
            # Motion detection sensitivity
            min_area = st.slider(
                "📐 Motion Sensitivity",
                min_value=100, max_value=5000,
                value=config.get("motion_detection", {}).get("min_contour_area", 500),
                step=100,
                help="Minimum contour area to trigger motion detection (lower = more sensitive)"
            )
            config.setdefault("motion_detection", {})["min_contour_area"] = min_area
            
            # Gaussian blur kernel
            blur_kernel = st.slider(
                "🔮 Blur Kernel Size",
                min_value=1, max_value=15,
                value=config.get("preprocessing", {}).get("gaussian_blur_kernel", 5),
                step=2,
                help="Gaussian blur kernel size (must be odd)"
            )
            config.setdefault("preprocessing", {})["gaussian_blur_kernel"] = blur_kernel

            # Frame skipping (decimation)
            frame_skip = st.slider(
                "⏭️ Frame Skipping",
                min_value=1, max_value=10,
                value=config.get("preprocessing", {}).get("frame_skip", 3),
                step=1,
                help="Process every N-th frame. Higher values speed up analysis but may skip fast-moving events (1 = process every frame)"
            )
            config.setdefault("preprocessing", {})["frame_skip"] = frame_skip

            # Advanced Preprocessing Enhancements
            st.markdown("---")
            st.markdown("## 🔧 Image Enhancements")
            enable_contrast_stretching = st.checkbox(
                "📈 Contrast Stretching",
                value=config.get("preprocessing", {}).get("enable_contrast_stretching", False),
                help="Stretch the image intensity values to normalize the dynamic range"
            )
            config.setdefault("preprocessing", {})["enable_contrast_stretching"] = enable_contrast_stretching

            enable_sharpening = st.checkbox(
                "✨ Image Sharpening",
                value=config.get("preprocessing", {}).get("enable_sharpening", False),
                help="Sharpen the image borders to enhance edge detection accuracy"
            )
            config.setdefault("preprocessing", {})["enable_sharpening"] = enable_sharpening

            gamma = st.slider(
                "🔆 Gamma Correction",
                min_value=0.1, max_value=3.0,
                value=config.get("preprocessing", {}).get("gamma", 1.0),
                step=0.1,
                help="Power-law transformation. Gamma > 1.0 brightens shadows, < 1.0 darkens highlights."
            )
            config.setdefault("preprocessing", {})["gamma"] = gamma

            
            st.markdown("---")
            st.markdown("## 📊 Detection Classes")
            detect_persons = st.checkbox("👤 Detect Persons", value=True)
            detect_vehicles = st.checkbox("🚗 Detect Vehicles", value=True)
            detect_animals = st.checkbox("🐾 Detect Animals", value=True)
            
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; color: rgba(255,255,255,0.3); font-size: 0.75rem; padding: 1rem 0;">
                <p>Restricted Area Monitor v1.0</p>
                <p>Computer Vision Project</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; color: rgba(255,255,255,0.4); font-size: 0.85rem; padding: 1rem 0;">
                <p>🔑 Use this directory manager to add approved vehicles and personnel profiles to whitelist them from triggering security alerts.</p>
            </div>
            """, unsafe_allow_html=True)

    if app_page == "🔑 Authorized Directory Manager":
        render_authorized_directory_manager(config)
        return

    # Render header
    render_header()
    
    # ---- Initialize Session State ----
    if "video_path" not in st.session_state:
        st.session_state.video_path = None
    if "roi_points" not in st.session_state:
        st.session_state.roi_points = []
    if "processing_done" not in st.session_state:
        st.session_state.processing_done = False
    if "results" not in st.session_state:
        st.session_state.results = None
    
    # ============================================================
    # STEP 1: Upload Video
    # ============================================================
    render_step(1, "Upload CCTV Video Footage")
    
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv", "wmv"],
        help="Upload a CCTV/surveillance video to analyze",
        key="video_uploader"
    )
    
    if uploaded_file is not None:
        # Save uploaded file to temp location
        temp_dir = os.path.join(os.path.dirname(__file__), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state.video_path = temp_path
        
        # Display video info with optional rotation
        video_info = get_video_info(temp_path)
        rotation_mode = config.get("preprocessing", {}).get("video_rotation", "0°")
        disp_width = video_info['width']
        disp_height = video_info['height']
        if rotation_mode in ["90° Clockwise", "90° Counter-Clockwise"]:
            disp_width, disp_height = disp_height, disp_width
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_metric_card("🎬", f"{video_info['duration']:.1f}s", "Duration")
        with col2:
            render_metric_card("📐", f"{disp_width}×{disp_height}", "Resolution")
        with col3:
            render_metric_card("🎞️", video_info['frame_count'], "Total Frames")
        with col4:
            render_metric_card("⚡", f"{video_info['fps']:.0f}", "FPS")
        
        # Show video preview
        st.video(temp_path)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # ============================================================
        # STEP 2: Define Restricted Area
        # ============================================================
        render_step(2, "Draw the Restricted Area on the First Frame")
        
        st.info("🖊️ **Draw a polygon** on the image below to define the restricted area. Use the rectangle tool or freeform drawing to mark the zone you want to monitor.")
        
        # Extract and display first frame
        first_frame = extract_first_frame(temp_path)
        
        if first_frame is not None:
            # Apply rotation to first frame
            if rotation_mode == "90° Clockwise":
                first_frame = cv2.rotate(first_frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation_mode == "180°":
                first_frame = cv2.rotate(first_frame, cv2.ROTATE_180)
            elif rotation_mode == "90° Counter-Clockwise":
                first_frame = cv2.rotate(first_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
            # Convert BGR to RGB for display
            first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            frame_image = Image.fromarray(first_frame_rgb)
            
            # Calculate canvas dimensions (maintain aspect ratio, max width 700)
            canvas_width = min(700, disp_width)
            scale_factor = canvas_width / disp_width
            canvas_height = int(disp_height * scale_factor)
            
            # Resize frame for canvas background
            frame_resized = frame_image.resize((canvas_width, canvas_height))
            
            # Drawing mode selection
            drawing_mode = st.radio(
                "Drawing Tool:",
                ["rect", "freedraw", "polygon"],
                horizontal=True,
                help="Rectangle is easiest for simple areas. Use freedraw or polygon for complex shapes."
            )
            
            # Streamlit drawable canvas
            canvas_result = st_canvas(
                fill_color="rgba(0, 255, 0, 0.15)",
                stroke_width=3,
                stroke_color="#00ff00",
                background_image=frame_resized,
                height=canvas_height,
                width=canvas_width,
                drawing_mode=drawing_mode,
                key="roi_canvas",
            )
            
            # Extract ROI points from canvas
            roi_points = canvas_points_to_polygon(canvas_result)
            
            if roi_points:
                # Scale points back to original video dimensions
                scaled_points = [
                    (int(x / scale_factor), int(y / scale_factor))
                    for x, y in roi_points
                ]
                st.session_state.roi_points = scaled_points
                st.success(f"✅ Restricted area defined with {len(scaled_points)} points")
            
            # Also allow manual coordinate input
            with st.expander("📝 Or enter coordinates manually"):
                coord_input = st.text_area(
                    "Enter polygon coordinates (one per line, format: x,y)",
                    placeholder="100,100\n500,100\n500,400\n100,400",
                    help="Enter coordinates of the restricted area polygon"
                )
                if coord_input.strip():
                    try:
                        manual_points = []
                        for line in coord_input.strip().split("\n"):
                            x, y = line.strip().split(",")
                            manual_points.append((int(x.strip()), int(y.strip())))
                        st.session_state.roi_points = manual_points
                        st.success(f"✅ Manual coordinates loaded: {len(manual_points)} points")
                    except ValueError:
                        st.error("❌ Invalid format. Use 'x,y' per line.")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # ============================================================
        # STEP 3: Process Video
        # ============================================================
        render_step(3, "Process Video & Detect Unauthorized Entry")
        
        if st.session_state.roi_points:
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f"""
                <div class="processing-info">
                    <strong>📋 Ready to process:</strong><br>
                    • Video: {uploaded_file.name}<br>
                    • ROI: {len(st.session_state.roi_points)} polygon points<br>
                    • Confidence: {confidence}<br>
                    • Motion sensitivity: {min_area}px²
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                process_btn = st.button(
                    "🚀 Start Processing",
                    type="primary",
                    use_container_width=True
                )
            
            if process_btn:
                st.markdown("---")
                
                # Progress indicators
                progress_bar = st.progress(0, text="Initializing...")
                status_text = st.empty()
                preview_placeholder = st.empty()
                
                # Run the processing pipeline
                with st.spinner("Processing video..."):
                    decision_engine, key_frames, output_path = process_video(
                        st.session_state.video_path,
                        st.session_state.roi_points,
                        config,
                        progress_bar,
                        status_text,
                        preview_placeholder
                    )
                
                # Store results
                st.session_state.processing_done = True
                st.session_state.results = {
                    "decision_engine": decision_engine,
                    "key_frames": key_frames,
                    "output_path": output_path
                }
                
                st.rerun()
        else:
            st.warning("⚠️ Please draw the restricted area in Step 2 before processing.")
        
        # ============================================================
        # STEP 4: Results Display
        # ============================================================
        if st.session_state.processing_done and st.session_state.results:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            render_step(4, "Detection Results & Analysis")
            
            results = st.session_state.results
            engine = results["decision_engine"]
            key_frames = results["key_frames"]
            output_path = results["output_path"]
            
            summary = engine.get_summary()
            
            # ---- Summary Metrics ----
            st.markdown("### 📊 Detection Summary")
            
            col1, col2, col3 = st.columns(3)

            with col1:
                render_metric_card(
                    "🚗",
                    summary["total_detections"]["vehicles"],
                    "Vehicle Detections"
                )

            with col2:
                render_metric_card(
                    "🚘",
                    summary["total_detections"]["unique_vehicles"],
                    "Unique Vehicles"
                )

            with col3:
                render_metric_card(
                    "🐾",
                    summary["total_detections"]["animals"],
                    "Animals Detected"
                )
            st.markdown("")
            
            # Alert count metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                render_metric_card("🔔", summary["total_alerts"], "Total Alerts")
            with col2:
                render_metric_card("🔴", summary["alerts_by_severity"].get("CRITICAL", 0) + summary["alerts_by_severity"].get("HIGH", 0), "High Priority")
            with col3:
                render_metric_card("🟡", summary["alerts_by_severity"].get("MEDIUM", 0), "Medium Priority")
            with col4:
                render_metric_card("🟢", summary["alerts_by_severity"].get("LOW", 0), "Low Priority")
            
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            
            # ---- Tabs for Different Views ----
            tab1, tab2, tab3, tab4 = st.tabs([
                "🔔 Alerts", "🖼️ Key Frames", "📈 Analytics", "🎬 Output Video"
            ])
            
            
            # ---- Tab 1: Alerts ----
            with tab1:
                st.markdown("### Alert History")

                if engine.alerts:
                    for alert in engine.alerts:
                        render_alert_card(alert)
                else:
                    st.info(
                        "No alerts were generated. "
                        "The restricted area appears to be clear."
                    )

                # Allow CSV export even when there are zero alerts.
                if st.button("Export Alerts to CSV"):
                    csv_path = os.path.join("output", "alerts.csv")
                    engine.export_alerts_csv(csv_path)
                    st.success(f"Alerts exported to {csv_path}")
            
            # ---- Tab 2: Key Frames ----
            with tab2:
                st.markdown("### Frames with Detections")
                if key_frames:
                    # Pagination / view selection settings
                    total_kfs = len(key_frames)
                    col_sz, col_pg = st.columns([1, 2])
                    
                    with col_sz:
                        page_size_option = st.selectbox(
                            "Frames per page",
                            options=["20", "50", "100", "All"],
                            index=0,
                            key="keyframe_page_size"
                        )
                    
                    if page_size_option == "All":
                        limit_per_page = total_kfs
                    else:
                        limit_per_page = int(page_size_option)
                        
                    import math
                    num_pages = math.ceil(total_kfs / limit_per_page)
                    
                    kfs_to_show = key_frames
                    start_idx = 0
                    end_idx = total_kfs
                    
                    if num_pages > 1:
                        with col_pg:
                            page = st.number_input(
                                f"Page (1-{num_pages})",
                                min_value=1,
                                max_value=num_pages,
                                value=1,
                                step=1,
                                key="keyframe_page_input"
                            )
                        start_idx = (page - 1) * limit_per_page
                        end_idx = min(start_idx + limit_per_page, total_kfs)
                        st.info(f"Showing frames {start_idx + 1} to {end_idx} of {total_kfs}")
                        kfs_to_show = key_frames[start_idx:end_idx]
                    else:
                        st.info(f"Showing all {total_kfs} detection frames")
                        kfs_to_show = key_frames
                    
                    # Show key frames in a grid
                    for i, fr in enumerate(kfs_to_show):
                        if fr.annotated_frame is not None:
                            # Convert BGR to RGB
                            display_frame = cv2.cvtColor(fr.annotated_frame, cv2.COLOR_BGR2RGB)
                            
                            col_img, col_info = st.columns([2, 1])
                            with col_img:
                                st.image(display_frame, caption=f"Frame #{fr.frame_index}",
                                        use_container_width=True)
                            with col_info:
                                st.markdown(f"**Frame #{fr.frame_index}**")
                                if fr.persons:
                                    for p in fr.persons:
                                        dist_info = f" (dist: {p.face_distance:.2f})" if p.face_distance is not None else ""
                                        if p.is_authorized:
                                            face_status = f"🟢 Authorized: {p.identity}{dist_info}"
                                        else:
                                            face_status = f"✅ Face detected{dist_info}" if p.face_detected else "❌ No face"
                                        st.markdown(f"👤 Person ({p.detection.confidence:.0%}) — {face_status}")
                                        if p.face_image is not None:
                                            face_rgb = cv2.cvtColor(p.face_image, cv2.COLOR_BGR2RGB)
                                            caption = f"{p.identity if p.is_authorized else 'Face'}{dist_info}"
                                            st.image(face_rgb, caption=caption, width=120)
                                if fr.vehicles:
                                    for v in fr.vehicles:
                                        if v.is_authorized:
                                            plate_status = f"🟢 Authorized: {v.identity}"
                                        else:
                                            plate_status = f"🔢 Plate: {v.plate_text}" if v.plate_detected else "❌ No plate"
                                        st.markdown(f"🚗 {v.detection.class_label.capitalize()} ({v.detection.confidence:.0%}) — {plate_status}")
                                        if v.plate_image is not None:
                                            plate_rgb = cv2.cvtColor(v.plate_image, cv2.COLOR_BGR2RGB)
                                            caption = f"Approved Plate: {v.identity}" if v.is_authorized else "Number Plate"
                                            st.image(plate_rgb, caption=caption, width=150)
                                if fr.animals:
                                    for a in fr.animals:
                                        st.markdown(f"🐾 {a.animal_type} ({a.category}) — Risk: {a.risk_level}")
                            
                            st.markdown("---")
                else:
                    st.info("No detections were found in the restricted area.")
            
            # ---- Tab 3: Analytics ----
            with tab3:
                st.markdown("### Detection Analytics")
                
                # Detection type distribution pie chart
                det_data = summary["total_detections"]
                if det_data["total"] > 0:
                    col_chart1, col_chart2 = st.columns(2)
                    
                    with col_chart1:
                        fig_pie = px.pie(
                            values=[det_data["persons"], det_data["vehicles"], det_data["animals"]],
                            names=["Persons", "Vehicles", "Animals"],
                            title="Detection Distribution by Type",
                            color_discrete_sequence=["#ef4444", "#f59e0b", "#22c55e"],
                            hole=0.4
                        )
                        fig_pie.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font_color="rgba(255,255,255,0.8)"
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col_chart2:
                        # Alert severity distribution
                        sev_data = summary["alerts_by_severity"]
                        fig_bar = px.bar(
                            x=list(sev_data.keys()),
                            y=list(sev_data.values()),
                            title="Alerts by Severity",
                            color=list(sev_data.keys()),
                            color_discrete_map={
                                "LOW": "#22c55e", "MEDIUM": "#f59e0b",
                                "HIGH": "#ef4444", "CRITICAL": "#dc2626"
                            }
                        )
                        fig_bar.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font_color="rgba(255,255,255,0.8)",
                            showlegend=False,
                            xaxis_title="Severity",
                            yaxis_title="Count"
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # Detection timeline
                    if key_frames:
                        timeline_data = []
                        for fr in key_frames:
                            for p in fr.persons:
                                timeline_data.append({
                                    "Frame": fr.frame_index,
                                    "Type": "Person",
                                    "Confidence": p.detection.confidence
                                })
                            for v in fr.vehicles:
                                timeline_data.append({
                                    "Frame": fr.frame_index,
                                    "Type": "Vehicle",
                                    "Confidence": v.detection.confidence
                                })
                            for a in fr.animals:
                                timeline_data.append({
                                    "Frame": fr.frame_index,
                                    "Type": "Animal",
                                    "Confidence": a.detection.confidence
                                })
                        
                        if timeline_data:
                            df_timeline = pd.DataFrame(timeline_data)
                            fig_timeline = px.scatter(
                                df_timeline, x="Frame", y="Confidence",
                                color="Type", title="Detection Timeline",
                                color_discrete_map={
                                    "Person": "#ef4444",
                                    "Vehicle": "#f59e0b",
                                    "Animal": "#22c55e"
                                },
                                size="Confidence",
                                size_max=12
                            )
                            fig_timeline.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font_color="rgba(255,255,255,0.8)"
                            )
                            st.plotly_chart(fig_timeline, use_container_width=True)
                else:
                    st.info("No detection data available for analytics.")
            
            # ---- Tab 4: Output Video ----
            with tab4:
                st.markdown("### Annotated Output Video")
                if os.path.exists(output_path):
                    st.video(output_path)
                    
                    # Download button
                    with open(output_path, "rb") as vid_file:
                        st.download_button(
                            label="📥 Download Annotated Video",
                            data=vid_file,
                            file_name="annotated_detection.mp4",
                            mime="video/mp4"
                        )
                else:
                    st.warning("Output video file not found.")
    
    else:
        # No video uploaded yet - show instructions
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: rgba(255,255,255,0.4);">
            <p style="font-size: 3rem;">📹</p>
            <p style="font-size: 1.2rem; font-weight: 500;">Upload a CCTV video to get started</p>
            <p style="font-size: 0.9rem;">Supported formats: MP4, AVI, MOV, MKV, WMV</p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()
