from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLO model once
model = YOLO("yolov8n.pt")

def is_seed_planting(image_path):
    # Run detection
    results = model(image_path, conf=0.25)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return False, "No hand detected"

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return False, "Invalid image"

    h, w, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ---- Soil detection (brown colors) ----
    lower_brown = (10, 50, 20)
    upper_brown = (30, 255, 200)
    soil_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    soil_pixels = cv2.countNonZero(soil_mask)

    if soil_pixels < 15000:
        return False, "Background is not soil"

    # ---- Get FIRST detected object as hand/person proxy ----
    box = boxes.xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, box)

    # Safety clamp
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    hand_region = img[y1:y2, x1:x2]

    if hand_region.size == 0:
        return False, "Invalid hand region"

    # ---- Seed detection INSIDE hand only ----
    gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    seed_pixels = cv2.countNonZero(thresh)

    if seed_pixels < 150:
        return False, "No visible seeds in hand"

    if seed_pixels > 10000:
        return False, "Object too large to be seeds"

    # ---- Spatial logic: hand must be ABOVE soil ----
    hand_center_y = (y1 + y2) // 2
    soil_y_positions = np.where(soil_mask > 0)[0]

    if len(soil_y_positions) == 0:
        return False, "Soil not clearly visible"

    soil_mean_y = int(np.mean(soil_y_positions))

    if hand_center_y > soil_mean_y:
        return False, "Hand is not above soil"

    # ---- ALL CONDITIONS PASSED ----
    return True, "Seed planting intent detected"
