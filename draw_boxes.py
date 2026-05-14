from ultralytics import YOLO
import cv2

# Model
model = YOLO("yolov8n.pt")

# Analiz
results = model("room.jpg")

# Görseli yükle
image = cv2.imread("room.jpg")

for result in results:

    for box in result.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        # Kutu çiz
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)

        # Yazı yaz
        cv2.putText(
            image,
            class_name,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            2
        )

# Kaydet
cv2.imwrite("output.jpg", image)

print("output.jpg oluşturuldu!")