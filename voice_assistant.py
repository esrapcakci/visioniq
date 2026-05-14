import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import cv2
from ultralytics import YOLO
from ollama import chat

# =========================
# SES KAYDI
# =========================

duration = 5  # saniye
sample_rate = 44100

print("Konuşmaya başla ")

audio = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=1
)

sd.wait()

write("question.wav", sample_rate, audio)

print("Ses kaydedildi!")

# =========================
# WHISPER
# =========================

model_whisper = whisper.load_model("base")

result = model_whisper.transcribe("question.wav")

question = result["text"]

print("\nSorun:")
print(question)

# =========================
# KAMERA
# =========================

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

cv2.imwrite("camera.jpg", frame)

cap.release()

# =========================
# YOLO
# =========================

model_yolo = YOLO("yolov8n.pt")

results = model_yolo("camera.jpg")

objects = []

for result in results:

    for box in result.boxes:

        class_id = int(box.cls[0])

        class_name = model_yolo.names[class_id]

        objects.append(class_name)

# =========================
# LLaVA
# =========================

response = chat(
    model='llava',
    messages=[
        {
            'role': 'user',
            'content': f"""
Sen Türkçe konuşan bir AI asistansın.

Bulunan objeler:
{', '.join(set(objects))}

Kullanıcının sorusu:
{question}

Kısa cevap ver.
""",
            'images': ['camera.jpg']
        }
    ]
)

print("\nAI Cevabı:\n")

print(response['message']['content'])