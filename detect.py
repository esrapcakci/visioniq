from ultralytics import YOLO

# Modeli yükle
model = YOLO("yolov8n.pt")

# Fotoğrafı analiz et
results = model("room.jpg")

# Tespit edilen objeleri yaz
for result in results:
    boxes = result.boxes

    for box in boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        print("Bulunan obje:", class_name)