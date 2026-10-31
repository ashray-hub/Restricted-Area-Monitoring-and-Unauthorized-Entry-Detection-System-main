# Step 1: Video Input & Preprocessing

This document describes the design, implementation, mathematical foundations, and configurations of the **Video Input & Preprocessing** stage.

---

## 💡 Overview

Surveillance footage from CCTV or IP cameras is often degraded by poor lighting, high-frequency noise (such as rain, snow, or wind-blown leaves), or compression artifacts. This module standardizes incoming frames to ensure downstream models (motion detection and YOLO object detection) operate under optimal visual conditions.

The preprocessing pipeline consists of the following sequential steps:
1. **Frame Resizing (INTER_AREA)** (Standardizes dimensions)
2. **Gaussian Blurring** (Removes high-frequency noise)
3. **Contrast Stretching** (Optional: normalizes pixel intensity range)
4. **CLAHE (Contrast Limited Adaptive Histogram Equalization)** (Boosts local contrast)
5. **Gamma Correction** (Optional: power-law transformation for shadow/highlight adjustment)
6. **Image Sharpening** (Optional: edge and border enhancement)

---

## 🧮 Mathematical Foundations & Core Techniques

### 1. Frame Resizing via Area Interpolation (`cv2.INTER_AREA`)
Standardizing frame dimensions (e.g., to $640 \times 480$ pixels) guarantees stable inference speeds for the YOLO model and consistent contour sizing for the background subtractor. 
- **Methodology**: When downsampling (decimating) an image, standard linear or nearest-neighbor interpolations introduce aliasing artifacts (moiré patterns) because they sub-sample pixels without blending.
- **Math**: Area interpolation (`cv2.INTER_AREA`) divides the source image into a grid corresponding to the target size and calculates the average intensity of all source pixels that fall within the bounds of a target pixel:
  $$I_{\text{target}}(x_t, y_t) = \frac{1}{A} \iint_{\text{pixel area}} I_{\text{source}}(x, y) \, dx \, dy$$
  where $A$ is the ratio of the source pixel area to the target pixel area. This behaves like a local box filter, producing smooth, aliasing-free results that preserve general shape structures.

---

### 2. Gaussian Blurring (Low-Pass Noise Filtering)
Surveillance cameras often capture high-frequency noise (e.g. wind-blown leaves, rain, or sensor grain in low-light conditions) that can trick motion detectors. 
- **Methodology**: The frame is convolved with a 2D Gaussian kernel, which acts as a low-pass filter, attenuating high-frequency noise.
- **Math**: The 2D Gaussian function is defined as:
  $$G(x, y) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$
  where $x$ and $y$ are the horizontal and vertical distances from the center of the kernel, and $\sigma$ is the standard deviation.
  
  For a $5 \times 5$ kernel, a discrete matrix is computed. It is then normalized so that the sum of all elements equals $1.0$ (to preserve overall image brightness):
  $$K_{\text{norm}}(i, j) = \frac{K(i, j)}{\sum_{m} \sum_{n} K(m, n)}$$
  The output pixel intensity is computed by sliding this kernel across the image and calculating the weighted sum of neighboring pixels:
  $$I_{\text{out}}(x, y) = \sum_{i=-2}^{2} \sum_{j=-2}^{2} I(x+i, y+j) \cdot K_{\text{norm}}(i, j)$$

---

### 3. Min-Max Contrast Stretching (Normalization)
- **Methodology**: When a video feed is washed out or underexposed, the pixel intensities occupy a narrow range of the possible $[0, 255]$ spectrum. Contrast stretching maps this narrow band back to the full range.
- **Math**: The min-max stretching formula is:
  $$I_{\text{out}}(x, y) = \left( I_{\text{in}}(x, y) - I_{\text{min}} \right) \cdot \frac{255}{I_{\text{max}} - I_{\text{min}}}$$
  where $I_{\text{min}}$ and $I_{\text{max}}$ are the minimum and maximum pixel intensities found across the entire frame. This is implemented efficiently in OpenCV using `cv2.normalize` with `cv2.NORM_MINMAX`.

---

### 4. CLAHE (Contrast Limited Adaptive Histogram Equalization)
Standard Global Histogram Equalization applies a single mapping to the entire image. This often over-amplifies noise in regions that are already very dark or very bright, creating unnatural halo effects.
- **Methodology**: CLAHE solves this by processing the image in local tiles and limiting the contrast amplification:
  1. **Color Space Conversion**: The frame is converted from BGR to **LAB color space**. The **L-channel (Lightness)** represents luminance, while **A and B** contain color information. Isolating the L-channel allows us to enhance contrast without shifting or distorting colors.
  2. **Tile Division**: The image is divided into small, non-overlapping tiles (typically $8 \times 8$ pixels).
  3. **Contrast Limiting**: A local histogram is calculated for each tile. If any histogram bin exceeds the configured `clipLimit` (e.g. 2.0), those excess pixels are clipped and redistributed uniformly across all bins before calculating the Cumulative Distribution Function (CDF):
     $$CDF(g) = \sum_{i=0}^{g} p(i)$$
     where $p(i)$ is the probability of a pixel having intensity $i$. The slope of the CDF dictates contrast gain; clipping the histogram bins limits this slope, preventing noise over-amplification.
  4. **Bilinear Interpolation**: Stitching tiles back together directly causes blocky edge artifacts. To prevent this, CLAHE calculates the final pixel mapping by taking a bilinear interpolation of the mapping values from the four nearest surrounding tiles.
  5. **Reconstruction**: The enhanced L-channel is merged back with the unmodified A and B channels and converted back to BGR.

---

### 5. Gamma Correction (Power-Law Transformation)
- **Methodology**: Used to correct non-linear brightness. Setting $\gamma > 1$ recovers details from deep shadows (brightens dark regions), while $\gamma < 1$ dampens bright glare highlights.
- **Math**: The power-law equation is:
  $$I_{\text{out}} = 255 \cdot \left(\frac{I_{\text{in}}}{255}\right)^{\frac{1}{\gamma}}$$
- **Optimization (Lookup Table)**: Computing fractional exponents for millions of pixels per frame ($640 \times 480 \times 3 \approx 921,600$ float power operations) is highly CPU-intensive. To achieve real-time performance, a 256-element **Lookup Table (LUT)** is pre-computed at initialization:
  $$LUT[i] = \text{round}\left(255 \cdot \left(\frac{i}{255}\right)^{\frac{1}{\gamma}}\right) \quad \forall i \in [0, 255]$$
  During frame processing, the conversion is reduced to a hardware-accelerated memory lookup using `cv2.LUT`.

---

### 6. Image Sharpening (Laplacian Filter)
- **Methodology**: Sharpens borders and edges to make fine details (such as facial contours or license plate characters) more legible for DeepFace and EasyOCR.
- **Math**: The second spatial derivative (Laplacian) measures intensity rate-of-change. Subtracting the Laplacian from the original image enhances edge transitions:
  $$I_{\text{sharp}} = I - k \cdot \nabla^2 I$$
  In discrete image processing, this is realized by convolving the frame with a composite high-pass 2D kernel filter using `cv2.filter2D`:
  $$K_{\text{sharp}} = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 5 & -1 \\ 0 & -1 & 0 \end{bmatrix}$$
  The central element ($5$) amplifies the original pixel, while the surrounding elements ($-1$) subtract the average intensity of adjacent pixels, creating high-contrast edge boundaries.

---

## 🛠️ Code Implementation Details

Implemented in [preprocessor.py](file:///d:/CV/src/preprocessor.py):

- **Constructor (`__init__`)**:
  - Initializes resizing bounds, Gaussian kernel size, and CLAHE parameters.
  - Generates the Gamma lookup table (if $\gamma \neq 1.0$).
  - Pre-creates the `cv2.createCLAHE` object to prevent reallocation overhead during runtime.
- **Inference (`preprocess`)**:
  - Executes each step in sequence: Resize $\rightarrow$ Gaussian Blur $\rightarrow$ Contrast Stretching (if enabled) $\rightarrow$ CLAHE (LAB space) $\rightarrow$ Gamma Correction (if enabled) $\rightarrow$ Sharpening (if enabled).

---

## ⚙️ Configuration Parameters (`config/settings.yaml`)

```yaml
preprocessing:
  resize_width: 640                  # Target width (px) for motion & YOLO detection
  resize_height: 480                 # Target height (px) for motion & YOLO detection
  gaussian_blur_kernel: 5            # Size of the Gaussian filter (must be odd)
  clahe_clip_limit: 2.0              # Contrast limit (higher = more contrast)
  clahe_tile_grid_size: 8            # Grid dimensions (e.g. 8x8 tiles)
  frame_skip: 3                      # Process every N-th frame to conserve CPU
  enable_contrast_stretching: false  # Normalizes pixel dynamic range
  enable_sharpening: false           # Emphasizes borders and edges
  gamma: 1.0                         # Gamma adjustment (1.0 = no correction)
```

---

## 📈 Dataset & Calibration Guidelines

This module uses **classical mathematical computer vision** and does not require neural network training. 

To calibrate these parameters for a specific surveillance setup:
- **Low-Light / Night-Vision Feeds**: Set `enable_contrast_stretching: true` and set `gamma` to `1.5` or `1.8` to brighten dark regions. Set `clahe_clip_limit` to `3.0` to pull out shapes from shadows.
- **High-Noise / Wind & Rain Feeds**: Increase `gaussian_blur_kernel` to `7` or `9` to smooth out larger noise textures and prevent false motion alarms.
- **ANPR / Plate Enhancement**: Set `enable_sharpening: true` to help extract alphanumeric borders for OCR.
