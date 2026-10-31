import cv2
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def trim_video_and_extract_face():
    src_video = os.path.join(project_root, "data", "face_demographics_walking.mp4")
    dest_video = os.path.join(project_root, "data", "face_demographics_short.mp4")
    faces_dir = os.path.join(project_root, "data", "authorized_faces")
    dest_face = os.path.join(faces_dir, "test_intel_short.jpg")
    
    if not os.path.exists(src_video):
        print(f"❌ Source video not found: {src_video}")
        return False
        
    cap = cv2.VideoCapture(src_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    # Define video writer for the trimmed clip
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(dest_video, fourcc, fps, (width, height))
    
    # We want to crop from frame 100 to 220 (about 4 seconds of footage)
    start_frame = 100
    end_frame = 220
    face_capture_frame = 180  # Frame where face is close and very clear
    
    frame_idx = 0
    face_img = None
    
    # We will use MediaPipe to get the exact face crop at the target frame
    from src.utils import load_config
    from src.face_detector import FaceDetector
    config_path = os.path.join(project_root, "config", "settings.yaml")
    config = load_config(config_path)
    face_detector = FaceDetector(config.get("face_detection", {}))
    
    print("🎬 Trimming video and extracting matching face...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        
        # Write frames in our target range
        if start_frame <= frame_idx <= end_frame:
            out_video.write(frame)
            
            # Extract face at the target frame
            if frame_idx == face_capture_frame:
                # We can run face detection on the whole frame
                # By defining a large bbox that covers the walking person
                # We'll search for the face in the middle region
                person_bbox = (int(width * 0.2), int(height * 0.1), int(width * 0.8), int(height * 0.9))
                person_result = face_detector.detect_face(frame, person_bbox)
                if person_result.face_detected and person_result.face_image is not None:
                    face_img = person_result.face_image
                    
    cap.release()
    out_video.release()
    face_detector.release()
    
    if face_img is not None:
        os.makedirs(faces_dir, exist_ok=True)
        
        # Clean up any other "test_user_intel" or "test_intel_short" files to avoid clutter
        for file in os.listdir(faces_dir):
            if file.startswith("test_user_intel") or file.startswith("test_intel_short"):
                try:
                    os.remove(os.path.join(faces_dir, file))
                except Exception:
                    pass
                    
        cv2.imwrite(dest_face, face_img)
        
        # Clear DeepFace cached representations pkl files
        pkl_files = [f for f in os.listdir(faces_dir) if f.endswith('.pkl')]
        for pkl in pkl_files:
            try:
                os.remove(os.path.join(faces_dir, pkl))
            except Exception:
                pass
                
        print(f"✅ Trimmed video saved to: {dest_video}")
        print(f"✅ High-res matching face saved to: {dest_face}")
        return True
    else:
        print("❌ Could not extract face at target frame.")
        return False

if __name__ == "__main__":
    trim_video_and_extract_face()
