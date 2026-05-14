import cv2
from ultralytics import YOLO

# YOLO modeli
model = YOLO("yolov8n.pt")

# Kamera aç
cap = cv2.VideoCapture(0)

print("Kamera açıldı 😄")
print("Çıkmak için Q bas.")

while True:

    # Kameradan görüntü al
    ret, frame = cap.read()

    if not ret:
        break

    # Detection
    results = model(frame)

    # Annotated frame
    annotated_frame = results[0].plot()

    # Göster
    cv2.imshow(
        "Vision Memory AI - Live",
        annotated_frame
    )

    # Q ile çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizlik
cap.release()
cv2.destroyAllWindows()