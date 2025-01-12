import cv2

cap = cv2.VideoCapture("cars.mp4")

# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2()
while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape

    # Extract Region of interest
    roi = frame[340: 720,500: 800]

    # 1. Object Detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 100:
            cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
    # Object detection from Stable camera
        object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)