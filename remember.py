import sys
import json
from ultralytics import YOLO

image_path = sys.argv[1]

model = YOLO("yolov8n.pt")

results = model(image_path)

memory = []

for result in results:

    width = result.orig_shape[1]
    height = result.orig_shape[0]

    for box in result.boxes:

        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        x1, y1, x2, y2 = box.xyxy[0]

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        if center_x < width / 3:
            horizontal = "sol"
        elif center_x < 2 * width / 3:
            horizontal = "orta"
        else:
            horizontal = "sağ"

        if center_y < height / 3:
            vertical = "üst"
        elif center_y < 2 * height / 3:
            vertical = "orta"
        else:
            vertical = "alt"

        memory.append({
            "object": class_name,
            "image": image_path,
            "location": f"{vertical} {horizontal}"
        })

# JSON kaydet
with open("memory.json", "w") as f:
    json.dump(memory, f, indent=4)

print("Hafıza kaydedildi!")