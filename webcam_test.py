#!/usr/bin/env python3
"""Quick webcam test — press q or Esc to quit."""

import sys
import cv2

cam = int(sys.argv[1]) if len(sys.argv) > 1 else 0
cap = cv2.VideoCapture(cam)
if not cap.isOpened():
    sys.exit(f"Cannot open camera {cam}")

print(f"Camera {cam} opened. Press q or Esc to quit.")
while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imshow(f"Webcam {cam}", frame)
    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
        break

cap.release()
cv2.destroyAllWindows()
