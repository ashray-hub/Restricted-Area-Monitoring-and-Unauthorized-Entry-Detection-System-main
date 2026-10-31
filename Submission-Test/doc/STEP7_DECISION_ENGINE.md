# Step 7: Decision Engine & Alert System

This document describes the design, implementation, and configurations of the **Decision Engine & Alert System** module.

---

## 💡 Overview

The **Decision Engine** acts as the central coordinator of the system. It gathers findings from all downstream analysis modules (YOLOv8, OpenCV Haar Cascade face detector, and EasyOCR plate reader), applies predefined security rules, filters out repetitive notifications via a cooldown window, and logs alerts.

Additionally, it controls **Alert Suppression** by checking whitelists. If an intruder is recognized as authorized, the security alarm is suppressed, and they are marked as a benign visitor.

---

## 🛠️ Code Implementation Details

Implemented in [decision_engine.py](file:///d:/CV/src/decision_engine.py):

### 1. Verification and Suppression Logic
For every processed frame, the engine receives a `FrameResult` containing the list of active detections inside the ROI. It evaluates the verification status:
- **Authorized (Benign Visitor)**: 
  - If a person matches a profile in `data/authorized_faces/` (is_authorized = True, with their identity resolved).
  - If a vehicle license plate matches a record in `config/authorized_plates.json` (is_authorized = True).
  - **Result**: The alert is **suppressed**. No alert is logged to the CSV file, no sound is played, and the dashboard displays a **green bounding box** with the label `[Name/Plate] (Authorized)`.
- **Unauthorized (Threat Intruder)**:
  - If a person is detected but has no matching face profile (or the distance exceeds the threshold).
  - If a vehicle is detected but the license plate is unrecognized (or EasyOCR failed to read it).
  - If an animal is detected.
  - **Result**: A **security alert is triggered**. It is logged to the dashboard alert history, written to `detection_log.csv`, drawn with a **red/orange bounding box**, and triggers a visual indicator.

---

### 2. Severity Level Mapping
Unsuppressed alerts are assigned severity levels based on the object type and risk profile:
- **Persons**: Assigned a severity level of `HIGH`. If face cascades failed to find a face, a warning descriptor `(Face Not Visible)` is appended.
- **Vehicles**: Assigned a severity level of `MEDIUM`. The recognized plate characters are appended to the alert if ANPR succeeded.
- **Animals**:
  - `LOW` for domestic animals (e.g. stray cat or dog, which are logged but pose no security threat).
  - `HIGH` or `CRITICAL` for wild or high-risk animals (e.g. bear, elephant) to warn of physical safety hazards.

---

### 3. Temporal Cooldown Control
To prevent flooding the logs with 24 redundant alerts per second as a target moves across the camera frame, the decision engine enforces a cooldown filter:
- It tracks the last alert timestamp for each category (`person`, `vehicle`, `animal`).
- If a new trigger occurs within `cooldown_seconds` (configured in `settings.yaml`) of the previous alert for that category, the alert is suppressed from the logs and dashboard alerts panel.
- Bounding boxes are still drawn on the video stream in real-time, but alerts are throttled.

---

### 4. Data Logging
Unsuppressed alerts are appended to a local CSV spreadsheet (`detection_log.csv`) in the format:
```text
Timestamp,FrameIndex,Class,Plate,Authorized,Identity,FaceDistance,Severity,CooldownSuppressed
2026-06-10 23:15:30,120,person,,False,,0.72,HIGH,False
2026-06-10 23:15:45,340,vehicle,KL07AB1234,True,KL07AB1234,,MEDIUM,True
```

---

## ⚙️ Configuration Parameters (`config/settings.yaml`)

```yaml
alerts:
  cooldown_seconds: 5            # Repeat alerts of the same category are suppressed for 5 seconds
  severity_levels:
    person: "HIGH"
    vehicle: "MEDIUM"
    animal_domestic: "LOW"
    animal_wild: "HIGH"
```

---

## 📈 Dataset & Calibration Guidelines

This module is **fully rule-based and deterministic**.
- It uses no machine learning parameters or datasets.
- Adjust `cooldown_seconds` to balance audit trail depth vs. noise:
  - **Slow-moving scenes**: Set to `10` or `15` seconds to prevent redundant alert lines.
  - **High-traffic gates**: Set to `2` or `3` seconds to capture consecutive distinct intruders.
