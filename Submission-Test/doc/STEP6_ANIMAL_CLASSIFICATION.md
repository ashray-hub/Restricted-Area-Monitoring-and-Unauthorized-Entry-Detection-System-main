# Step 6: Animal Classification & Risk Assessment

This document describes the design, implementation, and configurations of the **Animal Classification & Risk Assessment** module.

---

## 💡 Overview

surveillance zones near wild habitats, warehouses, or agricultural land often face threats from animal intrusions. When YOLOv8 detects an `animal` within the restricted zone, this module categorizes it and evaluates its threat level.

---

## 🛠️ Code Implementation details

Implemented in [animal_classifier.py](file:///d:/CV/src/animal_classifier.py):
- Runs when YOLOv8 returns a detection mapped to the `animal` category.
- **Mapping Logic**:
  - The classifier inspects the specific COCO label of the detection (e.g., `dog`, `cat`, `cow`, `bear`, `elephant`).
  - It cross-references this label with three predefined lists loaded from the configuration file:
    - `domestic_animals`: Triggers low-severity notifications (e.g. stray dogs or cats).
    - `wild_animals`: Triggers high-severity warnings.
    - `high_risk_animals`: Triggers critical/immediate alerts (e.g., predators or large animals like bears or elephants that pose safety hazards).
- Returns an `AnimalResult` data class containing the animal type, category (domestic vs. wild), and the calculated threat risk.

---

## ⚙️ Configuration Parameters (`config/settings.yaml`)

```yaml
animal_classification:
  # Map COCO classes to categorizations
  domestic_animals: ["cat", "dog"]
  wild_animals: ["bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]
  high_risk_animals: ["bear", "elephant"]
```

---

## 📈 Dataset & Training Information

This module uses the classification capabilities of the base YOLOv8 model (`yolov8n.pt`).
- **Dataset**: The classes (cat, dog, cow, bear, etc.) are already trained as part of YOLOv8's COCO weights.
- **Customization**:
  - If you need to detect local wild species not included in the standard COCO dataset (e.g. wild boars, leopards, deer), you should collect images of these target species and **train a custom YOLOv8 model** (as described in **[STEP3_OBJECT_DETECTION.md](file:///d:/CV/doc/STEP3_OBJECT_DETECTION.md)**).
  - Once trained, add their class IDs/names to the configuration settings list under the correct category to classify their risk level accordingly.
- **No separate model training is required** for the classifier itself, as it relies on mapping YOLO's outputs.
