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

def extract_face_from_video(video_filename):
    print(f"\n🔍 Scanning {video_filename} for a clear face...")
    video_path = os.path.join(project_root, "data", video_filename)
    config_path = os.path.join(project_root, "config", "settings.yaml")
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return False
        
    config = load_config(config_path)
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    preprocessor = FramePreprocessor(config.get("preprocessing", {}))
    motion_detector = MotionDetector(config.get("motion_detection", {}))
    
    # Large ROI covering the whole screen
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
    
    # We construct a FaceDetector but bypass DeepFace database checks for search
    face_detector = FaceDetector(config.get("face_detection", {}))
    
    frame_idx = 0
    extracted = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        # Process every frame for high chance of finding a face
        processed_frame = preprocessor.preprocess(frame)
        motion_detected, motion_mask, contours = motion_detector.detect_motion(processed_frame, roi_mask_preprocessed)
        
        if motion_detected or True:  # Force object detection
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
                        # Save the face image as reference profile
                        out_dir = os.path.join(project_root, "data", "authorized_faces")
                        os.makedirs(out_dir, exist_ok=True)
                        
                        out_path = os.path.join(out_dir, "test_user_1.jpg")
                        cv2.imwrite(out_path, person_result.face_image)
                        
                        print(f"🎉 Success! Found and extracted face in {video_filename}:")
                        print(f"  - Frame index: {frame_idx}")
                        print(f"  - Person bounding box: {bbox}")
                        print(f"  - Face bounding box: {person_result.face_bbox}")
                        print(f"  - Reference face saved to: {out_path}")
                        extracted = True
                        break
            if extracted:
                break
                
    cap.release()
    face_detector.release()
    return extracted

def main():
    # Attempt extraction from sample.mp4 first, then the HD video
    if not extract_face_from_video("sample.mp4"):
        print("Could not extract face from sample.mp4. Trying HD video...")
        if not extract_face_from_video("8321860-hd_1920_1080_30fps.mp4"):
            print("❌ Could not extract any face from either video file.")

if __name__ == "__main__":
    main()
