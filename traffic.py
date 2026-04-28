import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8s.pt")

# Vehicle classes (COCO dataset IDs)
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Load video
cap = cv2.VideoCapture("traffic.mp4")

# Previous values for smoothing
prev_count = 0
prev_density = 0.0

while True:
    ret, frame = cap.read()

    # Loop video if ended
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Resize frame
    frame = cv2.resize(frame, (960, 640))
    h, w = frame.shape[:2]

    # Define ROI (lower part of frame)
    y_start = int(h * 0.35)
    roi = frame[y_start:h, :]

    # Perform detection
    results = model(roi, conf=0.20, iou=0.6, verbose=False)

    vehicle_boxes = []
    total_box_area = 0

    # Process detections
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])

            if cls_id not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Convert to full-frame coordinates
            y1_full = y1 + y_start
            y2_full = y2 + y_start

            area = (x2 - x1) * (y2 - y1)

            # Ignore very small detections
            if area < 800:
                continue

            vehicle_boxes.append((x1, y1_full, x2, y2_full, cls_id))
            total_box_area += area

    # Vehicle count with smoothing
    raw_count = len(vehicle_boxes)
    vehicle_count = int(0.6 * prev_count + 0.4 * raw_count)
    prev_count = vehicle_count

    # Density calculation
    roi_area = (h - y_start) * w
    density = total_box_area / roi_area

    # Normalize and reduce sensitivity
    density = min(density, 1.0)
    density *= 0.6

    # Smooth density
    density = 0.6 * prev_density + 0.4 * density
    prev_density = density

    # Signal logic
    if vehicle_count > 30 and density > 0.15:
        signal = "RED - Heavy Traffic"
        route_msg = "Heavy Traffic - Take Alternate Route"
        color = (0, 0, 255)

    elif vehicle_count > 10 or density > 0.08:
        signal = "GREEN (45 sec)"
        route_msg = "Moderate Traffic"
        color = (0, 165, 255)

    else:
        signal = "GREEN (20 sec)"
        route_msg = "Traffic Clear"
        color = (0, 255, 0)

    # Traffic level classification
    if density > 0.15:
        level = "HIGH"
    elif density > 0.08:
        level = "MEDIUM"
    else:
        level = "LOW"

    # Draw ROI box
    cv2.rectangle(frame, (0, y_start), (w, h), (255, 255, 0), 2)
    cv2.putText(frame, "ROI (analysis zone)", (10, y_start - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Draw bounding boxes
    for x1, y1, x2, y2, cls_id in vehicle_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, VEHICLE_CLASSES[cls_id], (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # UI overlay panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (520, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # Display information
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)

    cv2.putText(frame, f"Density: {density:.2f}", (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(frame, f"Traffic Level: {level}", (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(frame, f"Signal: {signal}", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.putText(frame, route_msg, (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    cv2.putText(frame, "Real-Time AI Traffic Optimization", (20, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Show frame
    cv2.imshow("AI Adaptive Traffic Signal System", frame)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()