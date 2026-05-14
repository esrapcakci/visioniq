# VisionIQ

YOLOv8 + OCR + Akıllı Soru Cevap destekli görsel analiz sistemi.

##  Özellikler

| Modül | Açıklama |
|---|---|
| YOLO v8 | 80 sınıf nesne tespiti, çoklu ölçek |
| Nesne Rengi | HSV ile renk analizi |
| EasyOCR | Türkçe + İngilizce metin okuma |
| Kural Motoru | Soru tipine göre kesin cevap üretimi |
| Geçmiş Hafıza | JSON tabanlı analiz geçmişi |
| Kamera | Gerçek zamanlı kamera desteği |
| LLaVA | Opsiyonel multimodal model desteği |

---

##  Kurulum

```bash
# 1. Repoyu klonla
git clone https://github.com/esrapcakci/visioniq.git

# 2. Klasöre gir
cd visioniq

# 3. Bağımlılıkları yükle
pip install -r requirements.txt

# 4. Uygulamayı çalıştır
streamlit run app_streamlit.py
```

---

## LLaVA Kurulumu (Opsiyonel)

```bash
# Ollama kur
https://ollama.com

# Model indir
ollama pull llava
```

---

##  Gereksinimler

- Python >= 3.9
- streamlit
- ultralytics
- opencv-python
- pillow
- numpy
- easyocr
- ollama

---

##  Kullanım

1. Uygulamayı aç
2. Görsel yükle veya kamera kullan
3. Türkçe soru yaz

Örnek:

- "Kaç kişi var?"
- "Remote nerede?"
- "Baskın renk ne?"
- "Ne yazıyor?"
- "Masanın üzerinde ne var?"

4. Analiz Et butonuna bas

---

##  Proje Yapısı

```txt
├── app_streamlit.py
├── requirements.txt
├── memory.json
└── README.md
```

---

##  Kullanılan Teknolojiler

- Ultralytics YOLOv8
- EasyOCR
- Streamlit
- Ollama + LLaVA
