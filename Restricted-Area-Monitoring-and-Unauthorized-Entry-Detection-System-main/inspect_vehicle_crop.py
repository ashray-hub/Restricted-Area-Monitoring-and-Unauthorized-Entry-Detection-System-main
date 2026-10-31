import cv2
from src.object_detector import ObjectDetector
from src.preprocessor import FramePreprocessor
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
detections = detector.detect(processed, 209)

scale_x = frame.shape[1] / processed.shape[1]
scale_y = frame.shape[0] / processed.shape[0]

count = 0
for detection in detections:
    if detection.class_name != "vehicle":
        continue

    x1, y1, x2, y2 = detection.bbox
    x1, y1, x2, y2 = (
        int(x1 * scale_x), int(y1 * scale_y),
        int(x2 * scale_x), int(y2 * scale_y)
    )

    crop = frame[
        max(0, y1):min(frame.shape[0], y2),
        max(0, x1):min(frame.shape[1], x2)
    ]

    if crop.size:
        count += 1
        filename = f"temp/vehicle_crop_{count}.jpg"
        cv2.imwrite(filename, crop)
        print(f"Saved: {filename}; bbox={(x1, y1, x2, y2)}")

if count == 0:
    print("No vehicles detected in this frame.")
