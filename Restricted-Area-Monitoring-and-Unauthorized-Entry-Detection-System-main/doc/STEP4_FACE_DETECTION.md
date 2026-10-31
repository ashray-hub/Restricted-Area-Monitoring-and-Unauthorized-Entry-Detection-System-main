# Step 4: Face Detection in Surveillance

This document describes the design, implementation, mathematical foundations, and configurations of the **Face Detection** module.

---

## 💡 Overview

When a human intruder (classified as a `person` by YOLOv8) enters the restricted zone, the system triggers targeted face analysis. Rather than scanning the entire high-resolution frame (which is computationally expensive and prone to false positives), the system crops the person's bounding box and performs local face detection.

To ensure 100% offline reliability and eliminate heavy system dependencies that crash on Windows during download or encoding initialization, we replaced MediaPipe with a custom **local OpenCV Multi-Cascade Face Detector**.

---

## 🧮 Theoretical Foundations: Haar Cascade Classifiers

Proposed by Paul Viola and Michael Jones in 2001, the Haar Cascade framework is a classical machine learning method for object detection. It consists of four core concepts:

### 1. Haar-Like Features
Haar features are rectangular features that resemble edge, line, and four-rectangle configurations. They compute the difference between the sum of pixel intensities in adjacent rectangular regions:
$$\text{Feature Value} = \sum \text{Pixels}_{\text{black}} - \sum \text{Pixels}_{\text{white}}$$
For example, the region under the eyes is often darker than the cheeks, which can be captured by a horizontal two-rectangle feature.

### 2. Integral Images (Constant-Time Area Sums)
Evaluating hundreds of thousands of rectangular features at different scales is computationally prohibitive. Viola and Jones solved this by introducing the **Integral Image** ($II$):
$$II(x, y) = \sum_{x' \le x, y' \le y} I(x', y')$$
Once the integral image is constructed in $O(\text{Width} \times \text{Height})$ time, the sum of pixels in *any* rectangular region $D$ bounded by coordinates $(x_1, y_1)$ to $(x_2, y_2)$ can be calculated in exactly **4 array lookups** ($O(1)$ complexity):
$$\text{Sum}(D) = II(A) + II(D) - II(B) - II(C)$$
where $A, B, C, D$ represent the corners of the rectangle.

### 3. AdaBoost (Adaptive Boosting)
Out of millions of possible rectangular features in a sub-window, only a small fraction are discriminative. AdaBoost is used to select the most relevant features and train a series of **weak classifiers**. It combines them linearly to form a **strong classifier**:
$$F(x) = \text{sign}\left( \sum_{t=1}^{T} \alpha_t h_t(x) \right)$$
where $h_t(x)$ is a weak classifier and $\alpha_t$ is its weight.

### 4. Cascaded Classifier Architecture
Instead of evaluating all features on a sub-window, features are grouped into sequential **stages**. 
- If a sub-window fails to pass the very first stage (which contains only a few highly discriminative features), it is immediately rejected as a "non-face".
- If it passes, it moves to the next stage, which evaluates more features.
- A sub-window must pass all stages to be classified as a face. This cascading structure allows the algorithm to reject background regions in constant time, concentrating CPU resources on candidate face areas.

---

## 🔄 Cascaded Fallback Logic & Profile Flipping

Human faces in surveillance feeds appear at various angles (frontal, tilted, or profile). To detect faces under varying poses without using heavy deep learning models, we implement a **sequential multi-cascade search** inside `_detect_face_local_opencv`:

```
                 ┌────────────────────────────────┐
                 │       Cropped Person Box       │
                 └───────────────┬────────────────┘
                                 │
                                 ▼
                 ┌────────────────────────────────┐
                 │    1. Frontal Alt2 Cascade     │ ◄── High accuracy, low false positives
                 └───────────────┬────────────────┘
                                 ├─────────────────────────┐
                                 │ (No Face)               │ (Face Found)
                                 ▼                         ▼
                 ┌────────────────────────────────┐   ┌─────────┐
                 │   2. Frontal Default Cascade   │   │ Return  │
                 └───────────────┬────────────────┘   │  Face   │
                                 ├─────────────────┐  └─────────┘
                                 │ (No Face)       │ (Face Found)
                                 ▼                 ▼
                 ┌────────────────────────────────┐   ┌─────────┐
                 │   3. Profile Face (Right)      │   │ Return  │
                 └───────────────┬────────────────┘   │  Face   │
                                 ├─────────────────┐  └─────────┘
                                 │ (No Face)       │ (Face Found)
                                 ▼                 ▼
                 ┌────────────────────────────────┐   ┌─────────┐
                 │   4. Flipped Profile (Left)    │   │ Return  │
                 └───────────────┬────────────────┘   │  Face   │
                                 ├─────────────────┐  └─────────┘
                                 │ (No Face)       │ (Face Found)
                                 ▼                 ▼
                         ┌──────────────┐  ┌──────────────┐
                         │ Return None  │  │ Map Coords & │
                         │ (No Face)    │  │ Return Face  │
                         └──────────────┘  └──────────────┘
```

### Profile Asymmetry & Left-Profile Flipping
The default OpenCV profile cascade (`haarcascade_profileface.xml`) is asymmetric; it is trained to detect only profiles facing to the **right**.
To detect profiles facing to the **left**, we apply a horizontal flip:
1. Flip the grayscale person crop horizontally:
   $$\text{gray}_{\text{flipped}} = \text{cv2.flip}(\text{gray}, 1)$$
2. Run the profile face detector on $\text{gray}_{\text{flipped}}$.
3. If a face is detected at coordinates $(x_{\text{flip}}, y, w, h)$, we map the horizontal coordinate back to the original non-flipped space:
   $$x_{\text{orig}} = W_{\text{crop}} - x_{\text{flip}} - w$$
   where $W_{\text{crop}}$ is the width of the person crop.
4. Return the face bounding box $(x_{\text{orig}}, y, w, h)$.

---

## 📏 Bounding Box Coordinate Mapping & Padded Cropping

Once a face is detected in the local person crop, coordinates must be mapped back to the original high-resolution camera frame:
- **X Mapping**: $X_{\text{frame}} = X_{\text{person\_bbox\_start}} + X_{\text{face}}$
- **Y Mapping**: $Y_{\text{frame}} = Y_{\text{person\_bbox\_start}} + Y_{\text{face}}$

### 20% Context Padding
Standard face detection bounding boxes crop tightly around the eyes, nose, and mouth. However, facial verification models (like Facenet) perform significantly better when they have access to facial context (jawline, ears, hair).
To resolve this, we expand the bounding box by **20% on all sides** prior to cropping:
$$\text{pad}_x = \text{round}(w \times 0.20), \quad \text{pad}_y = \text{round}(h \times 0.20)$$
$$\text{Crop Bounds} = \left[ y_1 - \text{pad}_y, \ y_2 + \text{pad}_y, \ x_1 - \text{pad}_x, \ x_2 + \text{pad}_x \right]$$
These coordinates are clamped to the video frame dimensions to prevent out-of-bounds array slicing. This padded crop is what is passed directly to the DeepFace verification module.

---

## ⚙️ Configuration Parameters (`config/settings.yaml`)

```yaml
face_detection:
  scale_factor: 1.1              # How much the image size is reduced at each image scale (10% steps)
  min_neighbors: 4               # How many neighbors each candidate rectangle should have to retain it
  min_size: 30                   # Minimum face size in pixels (e.g. 30x30 px)
```

---

## 📈 Model Performance & Benefits
By switching from MediaPipe to local OpenCV cascades, the system gains:
1. **Zero Weight Downloads**: MediaPipe requires runtime C++ weights downloads which fail on closed networks or systems without proxy configuration. Haar Cascades are XML-based definitions bundled directly in OpenCV.
2. **Speed**: Haar cascades run in a few milliseconds on standard CPU threads.
3. **No Thread Lockups**: Standard deep-learning-based detectors sometimes conflict with Streamlit's web threads, causing UI lag. Classical cascades run deterministically without thread locking.
