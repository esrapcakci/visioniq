import cv2
import time
from ultralytics import YOLO

# YOLO modeli
model = YOLO("yolov8n.pt")

# Kamera
cap = cv2.VideoCapture(0)

# Önceki frame objeleri
previous_objects = set()

print("Scene monitor başladı...")
print("Çıkmak için Q bas.")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Detection
    results = model(frame)

    current_objects = set()

    # Objeleri çıkar
    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])

            class_name = model.names[class_id]

            current_objects.add(class_name)

    # Yeni objeler
    new_objects = current_objects - previous_objects

    # Kaybolan objeler
    missing_objects = previous_objects - current_objects

    # Bildirimler
    for obj in new_objects:
        print(f"[YENİ] {obj} göründü.")

    for obj in missing_objects:
        print(f"[KAYIP] {obj} artık görünmüyor.")

    # Annotated frame
    annotated_frame = results[0].plot()

    cv2.imshow(
        "Scene Monitor",
        annotated_frame
    )

    # Güncelle
    previous_objects = current_objects

    # Çıkış
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # CPU öldürmeyelim
    time.sleep(1)

cap.release()
cv2.destroyAllWindows()