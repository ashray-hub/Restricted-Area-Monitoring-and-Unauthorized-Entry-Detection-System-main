# 🔒 Restricted Area Monitoring & Unauthorized Entry Detection System

A Computer Vision-based intelligent surveillance system designed to monitor restricted zones in real-time or pre-recorded CCTV footage, identify unauthorized entry, and automatically run specialized detectors based on the threat actor category (**humans**, **vehicles**, and **animals**).

---

## 💡 Conceptual Ideas & Project Flow

The core design philosophy is to minimize resource consumption and avoid running heavy deep learning models continuously. The pipeline follows a **motion-triggered cascading model evaluation** flow:

```
                  ┌──────────────────────────────┐
                  │      CCTV Video Source       │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │      Frame Preprocessing     │
                  │   (Resize, Blur, Enhance)    │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │    Motion Detection in ROI   │  ◄── Defines Restricted Polygon
                  │    (Background Subtraction)  │
                  └──────────────┬───────────────┘
                                 │
                                 ├─────────────────────────┐
                                 │ (No Motion)             │ (Motion Detected)
                                 ▼                         ▼
                  ┌──────────────────────────────┐   ┌───────────────────────────┐
                  │      Skip Frame Analysis     │   │   YOLO Object Detection   │
                  └──────────────────────────────┘   └─────────────┬─────────────┘
                                                                   │
                                           ┌───────────────────────┼───────────────────────┐
                                           │ (Person)              │ (Vehicle)             │ (Animal)
                                           ▼                       ▼                       ▼
                               ┌───────────────────────┐ ┌───────────────────────┐ ┌───────────────────────┐
                               │    Face Detection     │ │    Plate Detection    │ │ Animal Classification │
                               │   (OpenCV Cascades)   │ │   & OCR (EasyOCR)     │ │   (Risk Assessment)   │
                               └───────────┬───────────┘ └───────────┬───────────┘ └───────────┬───────────┘
                                           │                         │                         │
                                           ▼                         ▼                         │
                               ┌───────────────────────┐ ┌───────────────────────┐             │
                               │  DeepFace Verification│ │  Whitelist Check      │             │
                               │  (Facenet Skip Mode)  │ │  (authorized_plates)  │             │
                               └───────────┬───────────┘ └───────────┬───────────┘             │
                                           │                         │                         │
                                           └───────────┬─────────────┘                         │
                                                       │                                       │
                                                       ▼                                       ▼
                                           ┌───────────────────────┐               ┌───────────────────────┐
                                           │    Is Authorized?     │               │   Determine Severity  │
                                           └─────┬───────────┬─────┘               │   (Low / Medium / High)│
                                                 │           │                     └───────────┬───────────┘
                                      (Yes)      │           │ (No)                            │
                                                 ▼           ▼                                 ▼
                                   ┌─────────────────┐   ┌─────────────────────────────────────┐
                                   │ Suppress Alert  │   │        Trigger Security Alert       │
                                   │ (Green BBox &   │   │        (Red/Orange BBox, Log        │
                                   │  Display Name)  │   │         Alert, Play Siren)          │
                                   └─────────────────┘   └─────────────────────────────────────┘
```

1. **Restricted Zone (ROI) Definition**: The user outlines the perimeter (polygon) on the CCTV frame. All detection tasks are constrained within this perimeter.
2. **Video Cleaning & Enhancement**: CCTV video is often noisy or poorly lit. The preprocessor cleans the frames using Gaussian Blurring and enhances contrast adaptive to local regions.
3. **Motion Pre-filtering**: Instead of running deep learning (YOLO) on every single frame, the system continuously analyzes the ROI using fast background subtraction. If no motion is detected, deep learning inference is skipped entirely, saving processing power.
4. **Cascaded Specialized Detectors**: Once motion is verified, YOLO detects and classifies candidate intruders. The system then branches based on the intruder type:
   - **Humans** trigger a face detection scan using local OpenCV Cascade Classifiers. If a face is detected, it is cropped and run through DeepFace facial verification against the authorized faces directory.
   - **Vehicles** trigger license plate extraction (Sobel/Canny edge-geometry analysis) and character recognition (EasyOCR).
   - **Animals** are cross-referenced with domestic/wild lists to determine ecological threat/risk levels.
5. **Intelligent Alerting & Suppression**: The decision engine compares face identities and license plates against local whitelists. 
   - **Authorized Entries** (e.g. registered employees, whitelisted cars) bypass the security alert engine. They are drawn with **green bounding boxes** on the stream, and their identity is clearly labelled on-screen to suppress false alarms.
   - **Unauthorized Entries** (unrecognized people or unknown vehicles) trigger immediate **security alerts** (with red/orange bounding boxes, alert logs, and optional visual warnings). A cooldown window prevents redundant logs for the same active intrusion.

---

## 🏗️ Module Architectures & Ideas

### 1. Frame Preprocessor (`src/preprocessor.py`)
Applies a 3-step optimization to improve detection accuracy:
- **Resizing**: Downscales frames to a standard size (e.g., 640x480) for unified processing and faster YOLO inference.
- **Gaussian Blur**: Eliminates high-frequency noise (e.g., wind-blown leaves, rain artifacts) that could cause false motion triggers.
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Converts BGR to LAB color space and equalizes the L-channel (luminance) to balance out shadow and glare, making intruders visible in poor lighting.

### 2. Motion Detector (`src/motion_detector.py`)
Implements adaptive background modeling:
- Uses `MOG2` or `KNN` background subtraction inside the scaled ROI.
- Thresholds foreground shadows and performs morphological operations (`Erosion` followed by `Dilation`) to remove tiny noise and stitch fragmented contours of a single moving body.
- Discards contours below a configurable pixel area threshold (`min_contour_area`).

### 3. ROI Manager (`src/roi_manager.py`)
Manages spatial constraints:
- Evaluates if the center of a detected object's bounding box resides inside the defined polygon boundary using `cv2.pointPolygonTest`.
- Translates coordinates between the preprocessed space (used for motion/YOLO) and the original video dimensions (used for visualization, face detection, and plate extraction).

### 4. Object Detector (`src/object_detector.py`)
- Employs YOLOv8 nano model to locate and classify objects.
- Maps COCO categories into three generalized indices: `person`, `vehicle`, and `animal`.

### 5. Face Detector (`src/face_detector.py`)
- Standard YOLO models do not locate fine facial structures. When a person is detected, their region is cropped and passed to our local OpenCV Multi-Cascade Face Detector.
- Applies a fallback search order (`haarcascade_frontalface_alt2` -> `haarcascade_frontalface_default` -> `haarcascade_profileface` (right profile) -> horizontally flipped crop for left profiles) to locate faces at any angle.
- Maps face coordinates back to the original full-frame space with a 20% padded margin.

### 6. Plate Detector (`src/plate_detector.py`)
- When a vehicle is identified, the car crop is processed using edge-detection (Sobel/Canny) and contour geometry analysis to locate potential license plates.
- The located plate is binarized and read using EasyOCR.

### 7. Animal Classifier (`src/animal_classifier.py`)
- Uses COCO labels to determine if an animal is domestic (low threat) or wild.
- Wild animals trigger high-severity warnings, while domestic pets trigger low-severity logs.

### 8. Decision & Alert Engine (`src/decision_engine.py`)
- Gathers findings from all modules, assigns severity classes (Low, Medium, High, Critical), tracks intrusion timestamps, and manages a cooldown log file.

### 9. Whitelist Verification & Authorization (`src/decision_engine.py` & `src/face_detector.py`)
- Sanitizes and compares license plate OCR readouts against `config/authorized_plates.json` (removing all spaces and symbols via regex).
- Runs custom cropped and padded face frames through DeepFace (using the `Facenet` model and `cosine` distance metric) with the detector backend bypassed (`detector_backend="skip"`) to verify identity against front-facing photo registrations under `data/authorized_faces/`.
- **Alert Suppression**: Marks whitelisted/authorized entries as verified to suppress security alerts and draw **green bounding boxes** with the matched name or plate. Unauthorized entries immediately trigger **high-severity security alerts** (drawn with **red/orange bounding boxes**).

---

## 📂 Project Structure

```
CV/
├── config/
│   ├── settings.yaml           # Central settings for all vision modules
│   └── authorized_plates.json  # Whitelisted license plate records (local JSON)
├── data/
│   ├── sample.mp4              # Test surveillance video
│   └── authorized_faces/       # Directory for whitelisted face profiles
├── doc/
│   ├── Computer_Vision_Proposal.pdf  # Project proposal report
│   ├── flow clart.jpeg         # Workflow diagram
│   ├── RUNNING.md              # Technical setup and run guide
│   ├── STUDIES_IMPLEMENTATION.md # Research alignment report (Studies 3, 4, 5)
│   ├── STEP1_PREPROCESSING.md  # Video enhancement details
│   ├── STEP2_MOTION_DETECTION.md # Background subtraction & ROI setup
│   ├── STEP3_OBJECT_DETECTION.md # YOLOv8 details & custom training guide
│   ├── STEP4_FACE_DETECTION.md  # OpenCV Cascades & fallback searching details
│   ├── STEP5_ANPR.md            # EasyOCR license plate reader & custom training
│   ├── STEP6_ANIMAL_CLASSIFICATION.md # Animal categorization guidelines
│   ├── STEP7_DECISION_ENGINE.md  # Alert rules & cooldown logic
│   └── STEP8_WHITELIST_AUTHORIZATION.md # Whitelisting & DeepFace details (Alert Suppression)
├── src/
│   ├── __init__.py             # Data models (Detection, Results)
│   ├── preprocessor.py         # Noise cleaning & CLAHE enhancement
│   ├── motion_detector.py      # Background subtraction (MOG2/KNN)
│   ├── roi_manager.py          # Polygon ROI mapping & point tests
│   ├── object_detector.py      # YOLOv8 object detector
│   ├── face_detector.py        # OpenCV cascade face detector
│   ├── plate_detector.py       # OCR-based license plate extractor
│   ├── animal_classifier.py    # Domestic/Wild risk categorizer
│   ├── decision_engine.py      # Severity rules & alerts logger
│   └── utils.py                # Drawing & filesystem helpers
├── tests/
│   ├── test_pipeline.py        # Integration test & sanity check script
│   └── test_authorization.py   # Whitelist verification & alert suppression unit tests
├── app.py                      # Streamlit interactive frontend
├── requirements.txt            # Package list
└── README.md                   # Core concepts & system design (This file)
```

---

## 🚀 Quick Start & Setup

For detailed instructions on configuration, datasets, and virtual environments, refer to the [RUNNING.md](doc/RUNNING.md) guide. Here is the quick-start guide to get up and running:

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```


### 2. Run Verification Tests

Run the authorization tests:

```bash
python -m pytest tests/test_authorization.py -v
```

Run the vehicle-entry event tests:

```bash
python -m pytest tests/test_vehicle_entry_events.py -v
```

Run the complete test suite:

```bash
python -m pytest tests -v
```

The current test suite contains 7 tests covering authorization, vehicle entry, exit and re-entry, duplicate-alert prevention, and plate-recognition alert updates.


### 3. Launch Streamlit Surveillance Dashboard
```bash
python -m streamlit run app.py
```
*Note: Face verification using `DeepFace` will automatically download `VGG-Face` model weights (approx. 580MB) on its first run.*
