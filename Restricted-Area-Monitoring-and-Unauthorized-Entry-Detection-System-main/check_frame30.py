import cv2
from pathlib import Path

video = Path("temp/single.mp4")
output = Path("temp/frame30.jpg")

cap = cv2.VideoCapture(str(video))
cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
success, frame = cap.read()
cap.release()

if success:
    cv2.imwrite(str(output), frame)
    print(f"SUCCESS: Frame saved to {output.resolve()}")
else:
    print(f"ERROR: Could not read frame 30 from {video}")
