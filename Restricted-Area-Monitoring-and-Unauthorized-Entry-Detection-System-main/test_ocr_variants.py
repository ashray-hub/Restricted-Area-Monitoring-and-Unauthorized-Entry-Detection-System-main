import cv2
import easyocr

path = "temp/plate_fallback_test.jpg"
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise SystemExit(f"Cannot load {path}")

reader = easyocr.Reader(["en"], gpu=False, verbose=False)

_, otsu = cv2.threshold(
    image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

adaptive = cv2.adaptiveThreshold(
    image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 31, 5
)

variants = {
    "Original grayscale": image,
    "Otsu threshold": otsu,
    "Adaptive threshold": adaptive,
}

allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

for name, candidate in variants.items():
    print(f"\n--- {name} ---")
    results = reader.readtext(
        candidate,
        detail=1,
        paragraph=False,
        allowlist=allowlist,
    )

    if not results:
        print("No text detected")
        continue

    for box, text, confidence in results:
        print(f"Text: {text} | Confidence: {confidence:.3f}")
