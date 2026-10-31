# 📚 Connection to Literature Review Studies

This document details how the three core research fields reviewed in your proposal (**Study 3**, **Study 4**, and **Study 5**) are realized and implemented in our **Restricted Area Monitoring & Unauthorized Entry Detection** codebase.

---

## 🔍 Study 3: ROI-Based Object Detection Optimization

### Proposal Concept
The proposal outlines that analyzing every single pixel across the entire video frame is computationally expensive and introduces background noise/false detections. The research recommends **ROI masking prior to object detection** to restrict deep learning processing to only the portions of the frame that contain active motion in the designated zone.

### Codebase Realization
We implemented this optimization using a two-stage cascaded pipeline:
1. **Pre-filtering with Background Subtraction**:
   - Instead of running YOLOv8 on every frame, [motion_detector.py](file:///d:/CV/src/motion_detector.py) runs a fast OpenCV background subtraction model (`MOG2` or `KNN`) inside the restricted ROI.
   - It generates a binary mask [roi_manager.py](file:///d:/CV/src/roi_manager.py) scaled to the preprocessed dimensions.
   - If **no motion** is detected within the ROI contour area threshold (`min_contour_area`), the system immediately skips the rest of the frame, bypassing YOLOv8 completely.
2. **Cascaded Deep Learning**:
   - YOLOv8 [object_detector.py](file:///d:/CV/src/object_detector.py) is triggered **only** when motion is confirmed, reducing CPU/GPU load significantly.
   - Bounding boxes are scaled back to the original dimensions to confirm if they reside inside the original ROI polygon via `is_inside_roi()`.

---

## 👤 Study 4: Face Recognition-Based Surveillance Systems

### Proposal Concept
The proposal reviews deep learning face recognition as a method to verify human identities in surveillance streams. Face recognition is triggered specifically when a human intruder enters a restricted area to distinguish authorized personnel from intruders.

### Codebase Realization
We implemented targeted face analysis in the human intrusion pipeline:
1. **Intruder Branching**:
   - If YOLO detects a `person` inside the restricted zone, the system triggers the face detection module [face_detector.py](file:///d:/CV/src/face_detector.py).
2. **OpenCV Cascade Face Detector**:
   - Rather than scanning the entire video frame for faces (which is slow and error-prone), we crop the person's bounding box region.
   - The cropped region is processed locally using a highly accurate, multi-cascade Haar classifier sequence (frontal alt2 $\rightarrow$ frontal default $\rightarrow$ right profile $\rightarrow$ flipped left profile).
3. **DeepFace Face Verification**:
   - The detected face is cropped with a 20% margin, bypassed in DeepFace's detector (`detector_backend="skip"`), and verified using the `Facenet` model.
4. **Keyframe Logging & Alert Suppression**:
   - If a match is found under our cosine distance threshold, the alert is suppressed, a green bounding box is drawn, and their identity is labeled. Unrecognized faces trigger security alerts (red bounding box).
   - In [app.py](file:///d:/CV/app.py), the detected face crops are shown in the **Key Frames** dashboard tab alongside timestamps for auditing.

---

## 🚗 Study 5: Automatic Number Plate Recognition (ANPR) Systems

### Proposal Concept
ANPR systems identify vehicles entering restricted areas by locating and reading characters on license plates. The proposal states that standard ANPR systems usually run independently, but it is highly beneficial to tie vehicle attribute classification (YOLO classification) together with OCR license plate validation.

### Codebase Realization
We implemented ANPR inside the vehicle intrusion pipeline:
1. **Vehicle Localization**:
   - When YOLO classifies an intruder as a `vehicle` (e.g. car, motorcycle, bus, truck), the vehicle bounding box is forwarded to [plate_detector.py](file:///d:/CV/src/plate_detector.py).
2. **License Plate Segmentation**:
   - Inside the vehicle crop, the system performs classical edge detection (Bilateral filter, Sobel/Canny) and contour geometry analysis to locate potential license plates.
   - The plate is cropped, enhanced using Otsu's binarization, and resized.
3. **EasyOCR Text Extraction**:
   - The enhanced plate region is read by the **EasyOCR Reader** engine (CRAFT + CRNN models).
   - The alphanumeric license characters are extracted and returned alongside a confidence score.
4. **Dashboard Auditing & Whitelisting**:
   - Character sequences are sanitized and matched against `config/authorized_plates.json` (suppressing alerts for whitelisted plates and drawing green bounding boxes, triggering alerts for unrecognized plates).
   - Plates are logged to `detection_log.csv` and displayed under the **Alerts** and **Key Frames** tabs in the dashboard.
