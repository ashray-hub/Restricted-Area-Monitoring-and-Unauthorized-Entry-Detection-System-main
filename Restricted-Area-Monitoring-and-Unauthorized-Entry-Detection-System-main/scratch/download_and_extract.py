import cv2
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils import load_config
from src.preprocessor import FramePreprocessor
from src.motion_detector import MotionDetector
from src.roi_manager import ROIManager
from src.object_detector import ObjectDetector
from src.face_detector import FaceDetector

def extract_best_face(video_path):
    print("\n🔍 Scanning the entire video to find the largest and clearest close-up face...")
    config_path = os.path.join(project_root, "config", "settings.yaml")
    
    config = load_config(config_path)
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    preprocessor = FramePreprocessor(config.get("preprocessing", {}))
    motion_detector = MotionDetector(config.get("motion_detection", {}))
    
    # ROI covering the entire video
    roi_points = [(0, 0), (width, 0), (width, height), (0, height)]
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
    face_detector = FaceDetector(config.get("face_detection", {}))
    
    frame_idx = 0
    best_face_image = None
    best_face_bbox = None
    best_face_area = 0
    best_frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        processed_frame = preprocessor.preprocess(frame)
        motion_detected, motion_mask, contours = motion_detector.detect_motion(processed_frame, roi_mask_preprocessed)
        
        # Run object detection
        detections = object_detector.detect(processed_frame, frame_idx)
        
        # Scale YOLO detections back to original dimensions
        scale_x_up = width / resize_width
        scale_y_up = height / resize_height
        
        for d in detections:
            if d.class_name == "person":
                x1, y1, x2, y2 = d.bbox
                bbox = (
                    int(x1 * scale_x_up),
                    int(y1 * scale_y_up),
                    int(x2 * scale_x_up),
                    int(y2 * scale_y_up)
                )
                
                # Run face detection
                person_result = face_detector.detect_face(frame, bbox)
                
                if person_result.face_detected and person_result.face_image is not None:
                    fw = person_result.face_bbox[2] - person_result.face_bbox[0]
                    fh = person_result.face_bbox[3] - person_result.face_bbox[1]
                    area = fw * fh
                    
                    # Track the largest face found
                    if area > best_face_area:
                        best_face_area = area
                        best_face_image = person_result.face_image.copy()
                        best_face_bbox = person_result.face_bbox
                        best_frame_idx = frame_idx
                        
    cap.release()
    face_detector.release()
    
    if best_face_image is not None:
        faces_dir = os.path.join(project_root, "data", "authorized_faces")
        os.makedirs(faces_dir, exist_ok=True)
        
        # Remove old test user images
        for file in os.listdir(faces_dir):
            if file.startswith("test_user_intel"):
                try:
                    os.remove(os.path.join(faces_dir, file))
                except Exception:
                    pass
                    
        out_path = os.path.join(faces_dir, "test_user_intel.jpg")
        cv2.imwrite(out_path, best_face_image)
        
        # Clear DeepFace cached representations pkl files
        pkl_files = [f for f in os.listdir(faces_dir) if f.endswith('.pkl')]
        for pkl in pkl_files:
            try:
                os.remove(os.path.join(faces_dir, pkl))
            except Exception:
                pass
                
        print(f"🎉 Success! Extracted the BEST (largest/closest) face:")
        print(f"  - Frame index: {best_frame_idx}")
        print(f"  - Face dimensions: {best_face_bbox[2]-best_face_bbox[0]}x{best_face_bbox[3]-best_face_bbox[1]} (Area: {best_face_area})")
        print(f"  - Reference face saved to: {out_path}")
        return True
    else:
        print("❌ No face detected in the entire video.")
        return False

def main():
    video_path = os.path.join(project_root, "data", "face_demographics_walking.mp4")
    if os.path.exists(video_path):
        if extract_best_face(video_path):
            print("\n🚀 Best face extraction setup complete!")
    else:
        print(f"❌ Video not found at {video_path}")

if __name__ == "__main__":
    main()
