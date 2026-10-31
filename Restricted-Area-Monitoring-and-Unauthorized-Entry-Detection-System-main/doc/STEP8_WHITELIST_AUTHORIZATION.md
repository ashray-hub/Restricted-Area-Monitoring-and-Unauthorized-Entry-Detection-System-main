# Step 8: Whitelist Verification & Authorization System

The authorization and verification system enables whitelisting of approved visitor vehicles and registered personnel. This module filters out benign movement (e.g. security patrols, delivery cars, employees) and prevents false alerts, allowing the dashboard to focus purely on actual unauthorized intrusions.

---

## 💡 System Flow Chart

The whitelisting pipeline sits directly between the specialized feature extractors and the alert generation engine:

```
                            ┌───────────────────────────────┐
                            │    YOLO Detection inside ROI  │
                            └───────────────┬───────────────┘
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │       Branch by Object Classification         │
                    └───────┬───────────────────────────────┬───────┘
                            │ (Person)                      │ (Vehicle)
                            ▼                               ▼
            ┌───────────────────────────────┐   ┌───────────────────────────────┐
            │   Crop Person & Detect Face   │   │   Locate Plate & Run EasyOCR  │
            │       (OpenCV Cascades)       │   │         (PlateDetector)       │
            └───────────────┬───────────────┘   └───────────────┬───────────────┘
                            │                               │
                            ▼                               ▼
            ┌───────────────────────────────┐   ┌───────────────────────────────┐
            │   DeepFace Face Verification  │   │   OCR String Sanitization &   │
            │   (Check data/authorized_faces/)  │   │   config/authorized_plates.json │
            └───────────────┬───────────────┘   └───────────────┬───────────────┘
                            │                               │
                            └───────────────┬───────────────┘
                                            │
                                            ▼
                            ┌───────────────────────────────┐
                            │    Is Whitelist Match?        │
                            └───────────────┬───────────────┘
                                            │
                                            ├─────────────────────────┐
                                            │ (Verified / Authorized) │ (Unverified / Unknown)
                                            ▼                         ▼
                            ┌───────────────────────────────┐   ┌───────────────────────────┐
                            │     Alert Suppressed          │   │   Trigger Security Alert  │
                            │ (Green BBox, display Identity)│   │ (Red/Orange BBox, Log CSV)│
                            └───────────────────────────────┘   └───────────────────────────┘
```

---

## 👤 Personnel Face Verification via DeepFace

For humans, the system runs facial verification using the **DeepFace** library:
1. **Reference Folder**: Reference front-facing photos of authorized people are stored under the folder `data/authorized_faces/`. The person's identity is extracted directly from the filename (e.g., `pakaya_1.jpg` $\rightarrow$ `"Pakaya 1"`, `ishan.jpg` $\rightarrow$ `"Ishan"`).
2. **DeepFace.find() Pipeline**:
   - The face region is detected in the person crop using OpenCV Cascades, expanded by a 20% margin, and cropped from the original high-resolution frame.
   - DeepFace builds facial embeddings using the `Facenet` model and compares them against the reference database.
   - **Bypassing Internal Detection**: We set `detector_backend="skip"` in `DeepFace.find()`. By default, DeepFace attempts to run its own face detectors (like MTCNN or RetinaFace) which require downloading heavy model weights (often crashing on Windows due to encoding/network issues). Since we already located and padded the face crop using Haar cascades, setting `"skip"` directs DeepFace to build embeddings directly from the cropped matrix, improving speed and reliability.
   - **Explicit Threshold Passing**: DeepFace uses strict internal default match thresholds for each model (e.g. cosine distance limit of $0.40$ for Facenet). If the calculated distance is $0.55$, DeepFace filters it out as a mismatch, ignoring any wider thresholds set in our code. To prevent this, we explicitly pass the threshold parameter:
     ```python
     dfs = DeepFace.find(
         img_path=face_image,
         db_path=self.authorized_faces_dir,
         model_name="Facenet",
         distance_metric="cosine",
         detector_backend="skip",
         threshold=self.threshold,    # <-- Critical fix: overrides strict defaults
         enforce_detection=False,
         silent=True
     )
     ```
     This ensures our configured threshold (default `0.65` for Facenet cosine distance) is respected.

3. **Embedding Comparison Math**:
   Facenet outputs a 128-dimensional embedding vector representing the face. The similarity between the query face embedding $\mathbf{A}$ and a database face embedding $\mathbf{B}$ is computed using **Cosine Distance**:
   $$D_C(\mathbf{A}, \mathbf{B}) = 1 - \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\|_2 \|\mathbf{B}\|_2}$$
   If $D_C \le 0.65$ (configured threshold), the face is marked as a match.

---

## 🔑 Vehicle Whitelisting & Normalization

The vehicle whitelisting logic matches OCR license plates against `config/authorized_plates.json`:
1. **Normalization**: EasyOCR readouts can introduce spacing or punctuation errors (e.g., `"KL-07 AB 1234"` instead of `"KL07AB1234"`). The system converts OCR readouts and whitelisted plates to uppercase and strips all non-alphanumeric characters using regular expressions:
   ```python
   clean_ocr = re.sub(r'[^A-Z0-9]', '', plate_text.upper())
   ```
2. **Suppression Check**: If a match is found in the cleaned array, `vehicle.is_authorized` is marked `True`, the vehicle's identity is assigned the matched plate string, and no security alert is generated.

---

## 🛑 Alert Suppression Rules (Authorized vs. Unauthorized)

The system enforces strict branching rules at the decision engine before visual rendering:

| Condition | Visual Indicator | Alert Logged? | Notification Generated? |
|-----------|------------------|---------------|-------------------------|
| **Authorized Person** | Green Bounding Box + `[Name] (Authorized)` | No | No (Suppressed) |
| **Authorized Vehicle** | Green Bounding Box + `[Plate] (Authorized)` | No | No (Suppressed) |
| **Unauthorized Person** | Red Bounding Box + `person (Unauthorized)` | **Yes** (Severity: HIGH) | **Yes** (Logged in CSV & Alerts Tab) |
| **Unauthorized Vehicle** | Red Bounding Box + `vehicle (Unauthorized)` | **Yes** (Severity: MEDIUM) | **Yes** (Logged in CSV & Alerts Tab) |
| **Animal** (Wild/High Risk) | Orange Bounding Box + `[Species] (Unauthorized)` | **Yes** (Severity: HIGH/CRIT) | **Yes** (Logged in CSV & Alerts Tab) |

---

## 🖥️ Directory Manager & Cache Invalidation

The Streamlit dashboard includes a dedicated management panel that allows adding/deleting authorized plates and uploading reference faces dynamically.
- **Dynamic Face Indexing**: DeepFace creates a pre-computed representation index file (e.g., `representations_facenet.pkl`) inside the authorized faces folder to speed up matches.
- **Cache Invalidation**: When a user uploads a new face profile or deletes an existing one via the Streamlit UI, the manager immediately deletes the `.pkl` files:
  ```python
  # Force DeepFace to re-compute representation embeddings on the next run
  for file in os.listdir(self.authorized_faces_dir):
      if file.endswith(".pkl"):
          os.remove(os.path.join(self.authorized_faces_dir, file))
  ```
  This forces DeepFace to index the new image on the next frame detection, ensuring the whitelist is always synchronized without restarting the application.
