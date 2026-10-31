import cv2
import easyocr

from src.object_detector import ObjectDetector
from src.preprocessor import FramePreprocessor
from src.plate_detector import PlateDetector
from src.utils import load_config

config = load_config("config/settings.yaml")

cap = cv2.VideoCapture("temp/single.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 209)
ok, frame = cap.read()
cap.release()

if not ok:
    raise SystemExit("Could not read frame 210.")

preprocessor = FramePreprocessor(config.get("preprocessing", {}))
processed = preprocessor.preprocess(frame)

detector = ObjectDetector(config.get("object_detection", {}))
plate_detector = PlateDetector(config.get("plate_detection", {}))

detections = detector.detect(processed, 209)

scale_x = frame.shape[1] / processed.shape[1]
scale_y = frame.shape[0] / processed.shape[0]

reader = None

for detection in detections:
    if detection.class_name != "vehicle":
        continue

    x1, y1, x2, y2 = detection.bbox
    x1, y1, x2, y2 = (
        int(x1 * scale_x), int(y1 * scale_y),
        int(x2 * scale_x), int(y2 * scale_y)
    )

    vehicle = frame[y1:y2, x1:x2]

    result = plate_detector.detect_plate(frame, (x1, y1, x2, y2))

    print("\nExisting detector:", result.plate_text)

    if result.plate_detected:
        continue

    h, w = vehicle.shape[:2]
    lower = vehicle[int(h * 0.72):int(h * 0.99),
                     int(w * 0.15):int(w * 0.85)]

    if lower.size == 0:
        print("Fallback crop is empty.")
        continue

    enlarged = cv2.resize(
        lower, None, fx=3, fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    cv2.imwrite("temp/plate_fallback_test.jpg", enlarged)

    if reader is None:
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    results = reader.readtext(enlarged)

    if results:
        for _, text, confidence in results:
            print(f"Fallback OCR: {text} | Confidence: {confidence:.3f}")
    else:
        print("Fallback OCR: No text found.")

print("\nDiagnostic complete. Main application was not changed.")
