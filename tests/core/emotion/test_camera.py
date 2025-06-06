import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera failed to open")
else:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from camera")
    else:
        print("Frame shape:", frame.shape)
cap.release()