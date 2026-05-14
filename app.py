import sys
from ultralytics import YOLO

image_path = sys.argv[1]
target_object = sys.argv[2].lower()

# YOLO modeli
model = YOLO("yolov8n.pt")

results = model(image_path)

found = False

for result in results:
    for box in result.boxes:

        class_id = int(box.cls[0])
        class_name = model.names[class_id].lower()

        if target_object in class_name:

            found = True

            x1, y1, x2, y2 = box.xyxy[0]

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Görsel boyutu
            width = result.orig_shape[1]
            height = result.orig_shape[0]

            # Yatay konum
            if center_x < width / 3:
                horizontal = "sol"
            elif center_x < 2 * width / 3:
                horizontal = "orta"
            else:
                horizontal = "sağ"

            # Dikey konum
            if center_y < height / 3:
                vertical = "üst"
            elif center_y < 2 * height / 3:
                vertical = "orta"
            else:
                vertical = "alt"

            print(f"\n{class_name} bulundu!")
            print(f"Konum: {vertical} {horizontal}")

if not found:
    print("\nObje bulunamadı.")