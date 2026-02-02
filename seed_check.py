from ultralytics import YOLO

# Load YOLO model once
model = YOLO("yolov8n.pt")

# Objects that strongly suggest planting
PLANTING_OBJECTS = {"sports ball", "apple", "orange"}

def check_planting(image_path):
    results = model(image_path, conf=0.25)
    detected = set()

    for r in results:
        for cls in r.boxes.cls:
            name = model.names[int(cls)]
            detected.add(name)

    # Decision logic
    planting = bool(detected & PLANTING_OBJECTS)

    return {
        "planting": planting,
        "objects": list(detected)
    }
