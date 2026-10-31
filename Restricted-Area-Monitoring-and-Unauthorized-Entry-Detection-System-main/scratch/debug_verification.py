import cv2
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils import load_config
from src.face_detector import FaceDetector

def main():
    print("[DEBUG] Starting face verification debugger...")
    video_path = os.path.join(project_root, "data", "face_demographics_walking.mp4")
    faces_dir = os.path.join(project_root, "data", "authorized_faces")
    
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found at {video_path}")
        return
        
    cap = cv2.VideoCapture(video_path)
    
    # We want to extract face at frame 90 (as shown in your screenshot)
    target_frame = 90
    frame_idx = 0
    frame_90 = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx == target_frame:
            frame_90 = frame.copy()
            break
    cap.release()
    
    if frame_90 is None:
        print("[ERROR] Could not read Frame 90 from video.")
        return
        
    # Initialize FaceDetector
    config_path = os.path.join(project_root, "config", "settings.yaml")
    config = load_config(config_path)
    face_detector = FaceDetector(
        config.get("face_detection", {}),
        config.get("authorization", {}),
        config.get("face_verification", {})
    )
    
    # Locate face in frame 90
    height, width = frame_90.shape[:2]
    person_bbox = (int(width * 0.2), int(height * 0.1), int(width * 0.8), int(height * 0.9))
    person_result = face_detector.detect_face(frame_90, person_bbox)
    
    if not person_result.face_detected or person_result.face_image is None:
        print("[ERROR] Face not detected in Frame 90 by MediaPipe!")
        return
        
    print(f"[OK] Face detected by MediaPipe. Bbox: {person_result.face_bbox}")
    
    # Now run DeepFace.find manually and print the results
    from deepface import DeepFace
    print("\n[INFO] Running DeepFace.find...")
    try:
        dfs = DeepFace.find(
            img_path=person_result.face_image,
            db_path=face_detector.authorized_faces_dir,
            model_name=face_detector.model_name,
            distance_metric=face_detector.distance_metric,
            detector_backend=face_detector.detector_backend,
            enforce_detection=False,
            silent=False
        )
        print("\n[INFO] DeepFace Find Output:")
        if dfs and len(dfs) > 0:
            df = dfs[0]
            print(df[['identity', 'distance', 'threshold', 'confidence']])
            if df.empty:
                print("[WARNING] Search completed but the dataframe is EMPTY (no matches found).")
        else:
            print("[WARNING] DeepFace returned no dataframes.")
    except Exception as e:
        print(f"[ERROR] DeepFace.find raised an exception: {e}")

if __name__ == "__main__":
    main()
