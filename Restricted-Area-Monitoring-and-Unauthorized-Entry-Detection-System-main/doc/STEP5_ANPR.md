# Step 5: Automatic Number Plate Recognition (ANPR)

This document describes the design, implementation, and configurations of the **ANPR (Automatic Number Plate Recognition)** module.

---

## 💡 Overview

When a vehicle is flagged inside the restricted zone, the system initiates ANPR to extract the license plate text. This allows for validation against an authorized vehicle list and logs plate entries for security auditing.

---

## 🛠️ Code Implementation details

Implemented in [plate_detector.py](file:///d:/CV/src/plate_detector.py):
1. **Vehicle Cropping**: Extracts the vehicle region from the original high-resolution frame.
2. **Classical License Plate Localization**:
   - Grayscale conversion → Bilateral Filter (reduces noise while preserving sharp edges).
   - Canny Edge Detection → Contour finding.
   - Filters contours based on aspect ratio (license plates are rectangular, typically having a width-to-height ratio between 2.0 and 6.0) and minimum/maximum area limits.
3. **Plate Extraction & Enhancement**:
   - Rotates or crops the plate area.
   - Applies Otsu's Thresholding (binarization) to maximize text contrast (black text on white background) and resizes the crop.
4. **OCR Character Recognition**:
   - Passes the enhanced crop to **EasyOCR**.
   - Filters recognized strings based on confidence.

---

## ⚙️ Configuration Parameters (`config/settings.yaml`)

```yaml
plate_detection:
  min_plate_width: 60            # Minimum width (pixels) of a plate inside the vehicle crop
  min_plate_height: 20           # Minimum height (pixels) of a plate
  ocr_languages: ["en"]          # EasyOCR language models to load
  ocr_confidence_threshold: 0.3  # Minimum confidence to accept OCR text
```

---

## 📈 Model Acquisition & Training Information

### How We Got the OCR Model
This module utilizes the pre-trained **EasyOCR** engine. EasyOCR uses deep learning models built on PyTorch:
- **Text Detection**: Uses a pre-trained **CRAFT** (Character Region Awareness for Text Detection) model.
- **Text Recognition**: Uses a **CRNN** (Convolutional Recurrent Neural Network) architecture composed of a ResNet feature extractor, BiLSTM sequence model, and CTC (Connectionist Temporal Classification) decoding layer.
- CRAFT and CRNN models are downloaded automatically on the first execution of `EasyOCR` and stored in `~/.EasyOCR/`.

---

### How to Train a Custom OCR Model
If you need to recognize custom font designs, non-standard symbols, or license plates from specific countries, you can retrain the text recognizer:

#### 1. Gather a Text Dataset
Collect cropped images of license plates along with a label mapping file:
```text
data/
├── train/
│   ├── image_001.jpg (contains plate text "ABC1234")
│   └── image_002.jpg (contains plate text "XYZ7890")
└── labels.txt
```
The `labels.txt` file format should be:
```text
train/image_001.jpg   ABC1234
train/image_002.jpg   XYZ7890
```

#### 2. Install the Training Toolkit
EasyOCR's text recognition is built on the standard **deep-text-recognition-benchmark** repository:
```bash
git clone https://github.com/clovaai/deep-text-recognition-benchmark.git
cd deep-text-recognition-benchmark
pip install -r requirements.txt
```

#### 3. Create LMDB Dataset files
The training script expects LMDB database formats. EasyOCR provides a converter script:
```bash
python create_lmdb_dataset.py --inputPath data/ --gtFile data/labels.txt --outputPath data_lmdb/
```

#### 4. Run CRNN Recognizer Training
Train the recognition model (ResNet + BiLSTM + CTC):
```bash
python train.py --train_data data_lmdb/ --valid_data val_lmdb/ --select_data / --batch_ratio 1 --Transformation None --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC --num_iter 50000
```
This saves custom weights (e.g. `best_accuracy.pth`).

#### 5. Integrate into EasyOCR
You can load your custom weights directly in EasyOCR:
```python
import easyocr
reader = easyocr.Reader(['en'], model_storage_directory='d:/CV/models', user_network_directory='d:/CV/user_network')
# Specify your custom network name (place weights under models/ and .yaml under user_network/)
result = reader.readtext(plate_crop, decoder='greedy', beamWidth=5)
```
*(For detailed configuration structure, consult the [EasyOCR Custom Model tutorial](https://github.com/JaidedAI/EasyOCR/blob/master/custom_model.md)).*
