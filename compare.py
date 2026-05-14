import sys
from ultralytics import YOLO

before_image = sys.argv[1]
after_image = sys.argv[2]

model = YOLO("yolov8n.pt")

# Konum hesaplama
def get_position(center_x, center_y, width, height):

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

    return f"{vertical} {horizontal}"

# Obje çıkarma
def detect_objects(image_path):

    results = model(image_path)

    detected = {}

    for result in results:

        width = result.orig_shape[1]
        height = result.orig_shape[0]

        for box in result.boxes:

            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            x1, y1, x2, y2 = box.xyxy[0]

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            position = get_position(
                center_x,
                center_y,
                width,
                height
            )

            detected[class_name] = position

    return detected

# Analiz
before_objects = detect_objects(before_image)
after_objects = detect_objects(after_image)

print("\n--- Değişim Analizi ---\n")

# Kaybolan objeler
for obj in before_objects:

    if obj not in after_objects:
        print(f"{obj} artık görünmüyor.")

# Yeni objeler
for obj in after_objects:

    if obj not in before_objects:
        print(f"Yeni obje bulundu: {obj}")

# Yer değiştirenler
for obj in before_objects:

    if obj in after_objects:

        old_pos = before_objects[obj]
        new_pos = after_objects[obj]

        if old_pos != new_pos:
            print(f"{obj} yer değiştirmiş.")
            print(f"Eski konum: {old_pos}")
            print(f"Yeni konum: {new_pos}")
            print()