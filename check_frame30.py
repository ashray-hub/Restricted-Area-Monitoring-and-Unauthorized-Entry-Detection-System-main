
import cv2
from pathlib import Path

video_path = Path("temp/single.mp4")
output_path = Path("temp/frame30.jpg")

cap = cv2.VideoCapture(str(video_path))
cap.set(cv2.CAP_PROP_POS_FRAMES, 30)

success, frame = cap.read()
cap.release()

if success:
    cv2.imwrite(str(output_path), frame)
    print(f"Saved frame to: {output_path}")
    print(f"Frame dimensions: {frame.shape[1]} x {frame.shape[0]}")
else:
    print("Could not read frame 30.")
