# Step 3: YOLOv8 Object Detection & Custom Model Training

This document details the object detection implementation and provides a complete guide on how to acquire and train a custom **YOLOv8** model for restricted area classification.

---

## 💡 Overview

When motion is detected in the ROI, the system triggers **YOLOv8** to locate and classify objects. YOLOv8 is a state-of-the-art, single-stage object detector that outputs bounding boxes and class probabilities in a single pass.

---

## 🛠️ Code Implementation details

Implemented in [object_detector.py](file:///d:/CV/src/object_detector.py):
- Loads the pre-trained weights file (e.g., `yolov8n.pt`).
- Maps COCO dataset classes to three project categories:
  - **`person`**: COCO class `0` (person)
  - **`vehicle`**: COCO classes `2` (car), `3` (motorcycle), `5` (bus), `7` (truck)
  - **`animal`**: COCO classes `14` to `23` (bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe)
- Bounding boxes are filtered using `confidence_threshold` (e.g., `0.5`) and Non-Maximum Suppression (NMS) `iou_threshold` (e.g., `0.45`).

---

## ⚙️ Configuration Parameters (`config/settings.yaml`)

```yaml
object_detection:
  model_path: "yolov8n.pt"       # Path to YOLO model weights (downloaded automatically if missing)
  confidence_threshold: 0.5      # Minimum confidence required to log an intruder
  iou_threshold: 0.45            # NMS IoU threshold
  device: "cpu"                  # "cpu" or "cuda" (for NVIDIA GPUs)
  person_classes: [0]
  vehicle_classes: [2, 3, 5, 7]
  animal_classes: [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
```

---

## 📈 Model Acquisition & Training Guide

### How We Got the Pre-trained Model
The system uses the default **YOLOv8 Nano** weights (`yolov8n.pt`) from Ultralytics. The model was trained on the **COCO (Common Objects in Context)** dataset, which contains 80 standard classes, covering all our base needs (people, standard cars/trucks, and common domestic/wild animals). The weights file is automatically downloaded on first run via the `ultralytics` package.

---

### How to Train a Custom YOLOv8 Model
If you need to optimize the detector for specific surveillance environments (e.g., thermal cameras, overhead angles, custom security uniforms, specific wildlife), you can train a custom model:

#### 1. Gather and Annotate a Dataset
Collect images of your restricted area containing target actors (humans, vehicles, specific animals).
- Use annotation tools like **Roboflow** (recommended), **CVAT**, or **LabelStudio**.
- Export the annotations in **YOLO format**. Each image will have a corresponding `.txt` file containing class indices and normalized bounding box coordinates:
  ```text
  <class_id> <x_center> <y_center> <width> <height>
  ```
  *(Example: `0 0.512 0.380 0.120 0.250` for a person centered in the middle).*

#### 2. Prepare the Directory Structure
Organize your dataset folder as follows:
```text
dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── dataset.yaml
```

#### 3. Define the Dataset Configuration (`dataset.yaml`)
Create a configuration file specifying the paths and the class names:
```yaml
path: d:/CV/dataset   # root dataset directory
train: train/images   # train images (relative to path)
val: val/images       # val images (relative to path)

# Classes: We map custom categories here
names:
  0: person
  1: vehicle
  2: animal
```

#### 4. Run the Training Script
Create a simple python training script `train.py`:
```python
from ultralytics import YOLO

def main():
    # 1. Load a model (load a pretrained Nano model for transfer learning)
    model = YOLO('yolov8n.pt')

    # 2. Train the model on your custom dataset
    results = model.train(
        data='d:/CV/dataset/dataset.yaml',
        epochs=100,            # Number of training epochs
        imgsz=640,             # Resolution size
        batch=16,              # Batch size (adjust based on GPU memory)
        device='gpu',          # Use 'gpu' (cuda) or 'cpu'
        workers=8,             # Number of CPU threads for data loading
        project='runs/train',  # Output folder
        name='custom_intruder_detector'
    )
    print("Training complete! Best weights saved to runs/train/custom_intruder_detector/weights/best.pt")

if __name__ == "__main__":
    main()
```

#### 5. Integrate the Custom Model
Once training completes, copy the `best.pt` file into your project directory and update `config/settings.yaml`:
```yaml
object_detection:
  model_path: "best.pt"
  person_classes: [0]     # Map according to your dataset.yaml indices
  vehicle_classes: [1]
  animal_classes: [2]
```
