# Step 2: ROI Management & Motion Detection

This document describes the design, implementation, and configurations of the **Region of Interest (ROI) and Motion Detection** stage.

---

## 💡 Overview

To achieve real-time efficiency, the system avoids running YOLOv8 object detection on every single frame. Instead, it uses a **fast, classic background subtraction** pre-filter to detect motion inside the user-defined restricted area. YOLO is only triggered when motion is confirmed within the ROI.

---

## 🛠️ Code Implementation details

### 1. ROI Polygon Masking ([roi_manager.py](file:///d:/CV/src/roi_manager.py))
- The user draws a polygon on the Streamlit dashboard canvas.
- These coordinates are mapped to a single-channel binary mask (`255` for inside the ROI, `0` for outside).
- Because motion and YOLO run on the preprocessed frame (`640x480`), the ROI points are scaled down to generate `roi_mask_preprocessed`.
- `cv2.pointPolygonTest` checks if the centers of detected YOLO bounding boxes reside inside the original ROI polygon boundaries.

### 2. Background Subtraction ([motion_detector.py](file:///d:/CV/src/motion_detector.py))
- The preprocessed frame is passed to the background subtractor (`MOG2` or `KNN`).
- The subtractor outputs a raw foreground mask representing change.
- **ROI Intersection**: The raw foreground mask is bitwise-ANDed with `roi_mask_preprocessed` to ignore any motion outside the restricted area (e.g. traffic on a public road in the background).
- **Morphological Cleanup**: An `Erosion` filter removes small pixel noise, and a subsequent double `Dilation` restores the shapes of moving objects.
- **Contour Filtering**: Contours are extracted from the cleaned mask. If any contour's area exceeds `min_contour_area` (configured in `settings.yaml`), the system flags that motion is present.

---

## ⚙️ Configuration Parameters (`config/settings.yaml`)

```yaml
motion_detection:
  algorithm: "MOG2"              # "MOG2" (faster) or "KNN" (more precise for slow movement)
  history: 500                   # Number of past frames used to build the background model
  var_threshold: 40              # Mahalanobis distance threshold to detect foreground
  detect_shadows: true           # Ignores shadows to prevent false motion triggers
  min_contour_area: 500          # Minimum moving object size in pixels to trigger detection
  learning_rate: -1              # Auto learning rate (-1) or manual rate (0 to 1)
```

---

## 📈 Dataset & Training Information

This step is **completely unsupervised**.
- The `MOG2`/`KNN` models learn the static background dynamically on-the-fly from the video stream itself.
- If a camera moves (e.g. due to wind) or the lighting changes suddenly, you can call `detector.reset()` to wipe the background cache and rebuild the model from scratch.
- **No training datasets are required**.
