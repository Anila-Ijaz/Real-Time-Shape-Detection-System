import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use larger model like yolov8m.pt for more accuracy

# Define shape detection function
def detect_shape(contour):
    approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
    num_sides = len(approx)
    if num_sides == 3:
        return "Triangle"
    elif num_sides == 4:
        return "Rectangle"
    elif num_sides > 4:
        return "Circle"
    return "Unknown"

# Main function
def analyze_frame(frame):
    results = model(frame)
    annotated = results[0].plot()

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])

        # Crop detected object
        cropped = frame[y1:y2, x1:x2]

        # Measure length and width in pixels
        length = y2 - y1
        width = x2 - x1

        # Shape detection using contours
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shape = "N/A"
        if contours:
            shape = detect_shape(contours[0])

        # Display all info (without color)
        info = f"{label}, {shape}"
        cv2.putText(annotated, info, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Fixed color

    return annotated

# Run from webcam or video
cap = cv2.VideoCapture(0)  # Use 0 for webcam or path to video file

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result_frame = analyze_frame(frame)
    cv2.imshow("Robot Arm Object Detection", result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
