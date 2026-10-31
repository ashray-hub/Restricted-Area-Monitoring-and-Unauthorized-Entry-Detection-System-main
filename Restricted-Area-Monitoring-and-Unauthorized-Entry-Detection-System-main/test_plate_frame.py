
import cv2

from src.object_detector import ObjectDetector
from src.plate_detector import PlateDetector
from src.utils import load_config
from src.preprocessor import FramePreprocessor

config = load_config("config/settings.yaml")

video_path = "temp/single.mp4"
frame_number = 209  # Frame 210

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
ok, frame = cap.read()
cap.release()

if not ok:
    print("Could not read frame.")
    raise SystemExit(1)

height, width = frame.shape[:2]

preprocessor = FramePreprocessor(config.get("preprocessing", {}))
processed_frame = preprocessor.preprocess(frame)

processed_height, processed_width = processed_frame.shape[:2]

scale_x = width / processed_width
scale_y = height / processed_height

object_detector = ObjectDetector(config.get("object_detection", {}))
plate_detector = PlateDetector(config.get("plate_detection", {}))

detections = object_detector.detect(processed_frame, frame_number)

print("Original frame:", width, "x", height)
print("Processed frame:", processed_width, "x", processed_height)
print("Vehicle detections:", sum(
    1 for d in detections if d.class_name == "vehicle"
))

for detection in detections:
    if detection.class_name != "vehicle":
        continue

    x1, y1, x2, y2 = detection.bbox

    original_bbox = (
        int(x1 * scale_x),
        int(y1 * scale_y),
        int(x2 * scale_x),
        int(y2 * scale_y),
    )

    print("\nVehicle bbox:", original_bbox)
    print("Confidence:", round(detection.confidence, 2))

    result = plate_detector.detect_plate(frame, original_bbox)

    print("Plate detected:", result.plate_detected)
    print("Plate text:", result.plate_text)
    print("Plate bbox:", result.plate_bbox)

