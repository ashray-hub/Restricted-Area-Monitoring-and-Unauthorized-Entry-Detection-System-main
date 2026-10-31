import cv2
import easyocr

image = cv2.imread("temp/vehicle_crop_1.jpg")

if image is None:
    raise SystemExit("Could not load vehicle crop.")

# Crop the lower part of the vehicle where the number plate appears.
h, w = image.shape[:2]
plate = image[int(h * 0.72):int(h * 0.99), int(w * 0.15):int(w * 0.85)]

# Enlarge the plate to help OCR.
plate = cv2.resize(plate, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

reader = easyocr.Reader(["en"], gpu=False, verbose=False)
results = reader.readtext(plate)

print("OCR results:")
for box, text, confidence in results:
    print(f"Text: {text} | Confidence: {confidence:.3f}")

cv2.imwrite("temp/plate_ocr_test.jpg", plate)
