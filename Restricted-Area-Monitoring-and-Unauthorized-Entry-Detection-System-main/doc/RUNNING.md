# 🛠️ System Setup, Technologies & Datasets Guide

This document provides step-by-step instructions on setting up, running, and testing the **Restricted Area Monitoring & Unauthorized Entry Detection System**, along with technical stack details and links to relevant datasets.

---

## 🛠️ Used Technologies

The system is built on a modern, high-performance Python computer vision stack:

| Component | Technology | Description |
|-----------|------------|-------------|
| **Core Language** | Python 3.11 | High-level programming language for the application logic |
| **Object Detection** | Ultralytics YOLOv8 | State-of-the-art detector for identifying persons, vehicles, and animals |
| **Face Detection** | OpenCV Haar Cascades | Local multi-cascade classifiers (`alt2`, `default`, `profile`, `flipped`) |
| **Number Plate OCR** | EasyOCR | Deep Learning-based Optical Character Recognition for plates |
| **Image Processing** | OpenCV | Used for frame extraction, resizing, drawing overlays, and color space conversions |
| **Motion Detection** | OpenCV Background Subtraction | Pre-filtered motion detection using MOG2/KNN algorithms |
| **Web Dashboard** | Streamlit | Responsive, custom-styled frontend for interactive monitoring and configuration |
| **Data & Charts** | Pandas & Plotly | Interactive analytics, charts, timeline graphs, and CSV logs |

---

## 🚀 Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.10+ installed on your system.

### 2. Install Dependencies
Open your command terminal, navigate to the project directory, and install the required libraries:
```bash
pip install -r requirements.txt
```

> [!NOTE]
> If you encounter dependency issues or binary warnings with TensorFlow on Windows, ensure that you are using the pinned versions in `requirements.txt` which install `tensorflow-cpu==2.13.0` to avoid conflicts.

---

## 🎬 How to Run the Application

### Running the Streamlit Dashboard
To launch the interactive web dashboard, run the following command in your terminal. This explicitly uses the virtual environment's Python interpreter to avoid any conflicts with global packages:
```bash
d:\CV\venv\Scripts\python.exe -m streamlit run app.py
```
*(If your virtual environment is located at a different path, adjust the prefix accordingly).*

After launching, the dashboard will automatically open in your default web browser at:
`http://localhost:8501`

### Running the Integration Test Script
To verify that all modules (motion, YOLOv8, OpenCV cascade face, EasyOCR plate, and animal classifier) work together correctly on your system, execute the sanity check test script:
```bash
python tests/test_pipeline.py
```

### Running Whitelist Verification Tests
To verify that the whitelisting, plate sanitization, and alert suppression engine work correctly under the dynamic verification rules, execute the authorization unit tests:
```bash
python tests/test_authorization.py
```

> [!WARNING]
> **DeepFace & TensorFlow Weights**: When running face verification for the first time, the `DeepFace` library will automatically download the pre-trained weights for the `Facenet` model (approx. 110MB) and store them locally under `~/.deepface/weights/`. Ensure you have an active internet connection on the first run.

---

## 📊 Dataset Links for Testing

To train, fine-tune, or test the models, we recommend the following public computer vision datasets:

1. **COCO Dataset (Common Objects in Context)**:
   - Includes standard labels for persons, vehicles, and various animals.
   - [COCO Dataset website](https://cocodataset.org/)
   - [Download COCO Annotations & Images](https://cocodataset.org/#download)

2. **VIRAT Video Dataset**:
   - High-quality realistic surveillance/CCTV footage containing pedestrian and vehicular activities.
   - [VIRAT Data website](https://viratdata.org/)

3. **Car License Plate Detection & OCR**:
   - Roboflow / Kaggle datasets containing vehicles and bounding boxes for license plates.
   - [Kaggle Car Plate Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection)
   - [Roboflow Universe License Plate Datasets](https://universe.roboflow.com/search?q=license+plate)

4. **Wildlife and Animal Detection**:
   - Specific animal datasets for classification testing.
   - [Kaggle Animals Detection Dataset](https://www.kaggle.com/datasets/gpiosenka/animals-detection-images-dataset)

5. **Sample CCTV Videos (Intel IoT Devkit)**:
   - Contains lightweight pre-recorded MP4 test videos.
   - [Intel IoT Sample Videos Repo](https://github.com/intel-iot-devkit/sample-videos)
   - [Direct download link for person-bicycle-car-detection.mp4](https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4)

6. **Free Stock CCTV Video Clips**:
   - High-quality, royalty-free surveillance camera style stock footage.
   - [Pexels Surveillance Video Clips](https://www.pexels.com/search/videos/surveillance/)
