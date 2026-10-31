"""
Face Detector Module
====================
Uses Google MediaPipe Face Detection to locate faces inside person
bounding boxes detected by YOLO.  Returns PersonResult data classes
with face coordinates mapped back to the original frame.

Usage:
    face_det = FaceDetector(config['face_detection'])
    person_result = face_det.detect_face(frame, person_bbox)
    face_det.release()
"""

from typing import Dict, Optional, Tuple
import cv2
import numpy as np
import os

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

from src import Detection, PersonResult


# ============================================================
# Face Detector Class
# ============================================================

class FaceDetector:
    """
    Detects faces inside cropped person regions using MediaPipe.

    The workflow:
        1. Crop the person region from the full frame.
        2. Run MediaPipe Face Detection on the crop.
        3. Convert face coordinates back to the original frame space.
        4. Return a PersonResult with the face image and metadata.

    Attributes:
        min_detection_confidence: Minimum detection confidence.
        model_selection: Model selection.
    """

    # Minimum crop size (pixels) below which face detection is skipped
    _MIN_CROP_SIZE: int = 30

    def __init__(self, config: Dict, auth_config: Dict = None, verification_config: Dict = None) -> None:
        """
        Initialise MediaPipe Face Detection and DeepFace configurations.

        Args:
            config: Dictionary from settings.yaml under 'face_detection'.
                    Expected keys: min_detection_confidence, model_selection.
            auth_config: Dictionary from settings.yaml under 'authorization'.
            verification_config: Dictionary from settings.yaml under 'face_verification'.
        """
        self.scale_factor: float = config.get("scale_factor", 1.1)
        self.min_neighbors: int = config.get("min_neighbors", 4)
        self.min_size: int = config.get("min_size", 30)

        # --- DeepFace verification settings ---
        auth_config = auth_config or {}
        verification_config = verification_config or {}
        
        # Resolve absolute path for self.authorized_faces_dir if it is relative
        faces_dir = auth_config.get("authorized_faces_dir", "data/authorized_faces")
        if faces_dir and not os.path.isabs(faces_dir):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.authorized_faces_dir = os.path.abspath(os.path.join(project_root, faces_dir))
        else:
            self.authorized_faces_dir = faces_dir
            
        self.model_name: str = verification_config.get("model_name", "Facenet")
        self.distance_metric: str = verification_config.get("distance_metric", "cosine")
        self.threshold: float = verification_config.get("threshold", 0.40)
        self.detector_backend: str = verification_config.get("detector_backend", "opencv")

        print(
            f"[FaceDetector] Initialised  |  "
            f"scale_factor={self.scale_factor}, "
            f"min_neighbors={self.min_neighbors}, "
            f"min_size={self.min_size}"
        )
        if DEEPFACE_AVAILABLE:
            print(
                f"[FaceDetector] DeepFace available | "
                f"db={self.authorized_faces_dir}, model={self.model_name}, "
                f"threshold={self.threshold}, backend={self.detector_backend}"
            )
        else:
            print("[FaceDetector] Warning: DeepFace not installed. Face verification will be skipped.")

    # ============================================================
    # Public Methods
    # ============================================================

    def _detect_face_local_opencv(self, person_crop: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect a face inside a person crop using OpenCV Haar Cascades
        (frontal_alt2 + frontal_default + profile + flipped profile).
        Returns the face bounding box (x, y, w, h) relative to the person crop, or None.
        """
        gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
        
        # Load classifiers (cached to avoid reloading them on every frame)
        if not hasattr(self, '_frontal_cascade_alt2'):
            self._frontal_cascade_alt2 = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
        if not hasattr(self, '_frontal_cascade_default'):
            self._frontal_cascade_default = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if not hasattr(self, '_profile_cascade'):
            self._profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
            
        candidates = []
        
        # 1. Frontal Alt2 (highly accurate, low false positive rate)
        if not self._frontal_cascade_alt2.empty():
            faces_alt2 = self._frontal_cascade_alt2.detectMultiScale(
                gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors, minSize=(self.min_size, self.min_size)
            )
            for f in faces_alt2:
                candidates.append((f, 'frontal_alt2'))
            
        # 2. Frontal Default (fallback)
        if not candidates and not self._frontal_cascade_default.empty():
            faces_def = self._frontal_cascade_default.detectMultiScale(
                gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors, minSize=(self.min_size, self.min_size)
            )
            for f in faces_def:
                candidates.append((f, 'frontal_default'))
                
        # 3. Profile (Right side)
        if not candidates and not self._profile_cascade.empty():
            faces_prof = self._profile_cascade.detectMultiScale(
                gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors, minSize=(self.min_size, self.min_size)
            )
            for f in faces_prof:
                candidates.append((f, 'profile_right'))
                
        # 4. Profile Flipped (Left side)
        if not candidates and not self._profile_cascade.empty():
            flipped_gray = cv2.flip(gray, 1)
            faces_prof_flipped = self._profile_cascade.detectMultiScale(
                flipped_gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors, minSize=(self.min_size, self.min_size)
            )
            for f in faces_prof_flipped:
                xf, y, w, h = f
                x_orig = gray.shape[1] - xf - w
                candidates.append(((x_orig, y, w, h), 'profile_left'))
                
        if not candidates:
            return None
            
        # Return candidate with the largest area (most likely to be the actual face)
        best_candidate = max(candidates, key=lambda c: c[0][2] * c[0][3])
        return best_candidate[0]

    def detect_face(
        self, frame: np.ndarray, person_bbox: Tuple[int, int, int, int]
    ) -> PersonResult:
        """
        Detect a face within a person bounding box.

        Steps:
            1. Crop the person region from the full frame.
            2. Run local OpenCV face detection on the crop (if not set to skip).
            3. Map the face bbox back to original frame coords.
            4. Crop the face image from the full frame with padding.
            5. Run DeepFace face verification on the face crop.

        Args:
            frame: Full BGR image as a numpy array (H, W, 3).
            person_bbox: Person bounding box as (x1, y1, x2, y2).

        Returns:
            PersonResult with face information populated. If no face is
            found, face_detected will be False and optional fields None.
        """
        px1, py1, px2, py2 = person_bbox

        # --- Build a placeholder Detection for PersonResult ---
        # (The caller will normally supply a real Detection; we only need
        #  the bbox here for coordinate mapping.)
        detection = Detection(
            bbox=person_bbox,
            class_name="person",
            class_label="person",
            confidence=0.0,
            frame_index=0,
        )

        # --- Validate and clamp person crop ---
        frame_h, frame_w = frame.shape[:2]
        px1 = max(0, px1)
        py1 = max(0, py1)
        px2 = min(frame_w, px2)
        py2 = min(frame_h, py2)

        crop_w = px2 - px1
        crop_h = py2 - py1

        # Skip if the crop is too small to contain a meaningful face
        if crop_w < self._MIN_CROP_SIZE or crop_h < self._MIN_CROP_SIZE:
            return PersonResult(detection=detection, face_detected=False)

        # --- Crop person region ---
        person_crop: np.ndarray = frame[py1:py2, px1:px2]

        # --- Run local OpenCV face detection ---
        face_image = None
        face_bbox = None

        if self.detector_backend == "skip":
            # Treat the entire person crop as the face
            face_bbox = (px1, py1, px2, py2)
            face_image = person_crop.copy()
        else:
            # Run our highly accurate local OpenCV multi-cascade face detector
            face_area = self._detect_face_local_opencv(person_crop)
            if face_area is not None:
                x, y, w, h = face_area
                
                # Convert to original frame coordinates
                face_x1 = int(px1 + x)
                face_y1 = int(py1 + y)
                face_x2 = int(px1 + x + w)
                face_y2 = int(py1 + y + h)
                
                # Clamp to frame boundaries
                face_x1 = max(0, face_x1)
                face_y1 = max(0, face_y1)
                face_x2 = min(frame_w, face_x2)
                face_y2 = min(frame_h, face_y2)
                
                face_bbox = (face_x1, face_y1, face_x2, face_y2)
                
                # Crop the face image from the full frame with padding
                if (face_x2 - face_x1) > 0 and (face_y2 - face_y1) > 0:
                    fw = face_x2 - face_x1
                    fh = face_y2 - face_y1
                    pad_x = int(fw * 0.20)
                    pad_y = int(fh * 0.20)
                    
                    x1_pad = max(0, face_x1 - pad_x)
                    y1_pad = max(0, face_y1 - pad_y)
                    x2_pad = min(frame_w, face_x2 + pad_x)
                    y2_pad = min(frame_h, face_y2 + pad_y)
                    
                    face_image = frame[y1_pad:y2_pad, x1_pad:x2_pad].copy()

        if face_image is None or face_bbox is None:
            return PersonResult(detection=detection, face_detected=False)

        # --- DeepFace verification ---
        is_authorized = False
        identity = None
        face_distance = None

        if face_image is not None and DEEPFACE_AVAILABLE:
            # Check if directory exists and contains images to prevent DeepFace crash
            has_images = False
            if self.authorized_faces_dir and os.path.exists(self.authorized_faces_dir):
                for file in os.listdir(self.authorized_faces_dir):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        has_images = True
                        break

            if has_images:
                try:
                    # DeepFace.find returns a list of dataframes
                    # Bypassing detection in DeepFace by setting detector_backend="skip"
                    dfs = DeepFace.find(
                        img_path=face_image,
                        db_path=self.authorized_faces_dir,
                        model_name=self.model_name,
                        distance_metric=self.distance_metric,
                        detector_backend="skip",
                        enforce_detection=False,
                        threshold=self.threshold,
                        silent=True
                    )
                    if dfs and len(dfs) > 0:
                        df = dfs[0]
                        if not df.empty:
                            best_match = df.iloc[0]
                            identity_path = best_match['identity']
                            dist_col = f"{self.model_name}_{self.distance_metric}"
                            
                            distance = None
                            for col in ['distance', dist_col]:
                                if col in best_match:
                                    distance = float(best_match[col])
                                    break
                                    
                            if distance is not None:
                                face_distance = distance
                                
                                import re
                                filename = os.path.basename(identity_path)
                                name_w_ext = os.path.splitext(filename)[0]
                                name_clean = re.sub(r'[\d_-]+$', '', name_w_ext)
                                name = name_clean.replace('_', ' ').title()
                                
                                if distance <= self.threshold:
                                    identity = name
                                    is_authorized = True
                                    print(f"[FaceDetector] Verified face: {identity} (dist: {distance:.4f} <= threshold: {self.threshold})")
                                else:
                                    print(f"[FaceDetector] Face match rejected: {name} (dist: {distance:.4f} > threshold: {self.threshold})")
                except Exception as e:
                    print(f"[FaceDetector] Verification error: {e}")

        return PersonResult(
            detection=detection,
            face_detected=True,
            face_bbox=face_bbox,
            face_image=face_image,
            is_authorized=is_authorized,
            identity=identity,
            face_distance=face_distance
        )

    def release(self) -> None:
        """
        Release face detector resources.
        """
        print("[FaceDetector] Resources released")
