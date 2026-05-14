import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import json
import os
import cv2
import numpy as np
from datetime import datetime
from collections import Counter

# ── Opsiyonel ────────────────────────────────────────────────
try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    from ollama import chat as ollama_chat
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="VQA Pro", page_icon="🔍", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: #0d0d0f; color: #e8e8e8; }
.main-header { font-family:'IBM Plex Mono',monospace; font-size:2rem; font-weight:600; color:#00ff9d; margin-bottom:.2rem; }
.sub-header  { font-size:.9rem; color:#666; margin-bottom:2rem; }
.section-title { font-family:'IBM Plex Mono',monospace; font-size:.75rem; color:#555;
    text-transform:uppercase; letter-spacing:.12em; margin:1.4rem 0 .5rem 0; }
.result-box { background:#111318; border:1px solid #1e2530; border-left:3px solid #00ff9d;
    border-radius:8px; padding:1.2rem 1.5rem; margin:.6rem 0; }
.obj-tag { display:inline-block; background:#0d2e1f; color:#00ff9d;
    border:1px solid #00ff9d30; border-radius:4px; padding:2px 10px; margin:3px;
    font-family:'IBM Plex Mono',monospace; font-size:.8rem; }
.ocr-tag { display:inline-block; background:#1a1a2e; color:#7eb8ff;
    border:1px solid #7eb8ff30; border-radius:4px; padding:2px 10px; margin:3px;
    font-family:'IBM Plex Mono',monospace; font-size:.8rem; }
.pos-row { background:#111318; border:1px solid #1e2530; border-radius:6px;
    padding:.55rem 1rem; margin:.25rem 0; font-family:'IBM Plex Mono',monospace;
    font-size:.82rem; color:#aaa; }
.ai-answer { background:#0a1a10; border:1px solid #00ff9d40; border-radius:10px;
    padding:1.4rem 1.6rem; font-size:1.15rem; color:#e8e8e8; line-height:1.7; margin-top:.8rem; }
.mem-card { background:#111318; border:1px solid #1e2530; border-radius:8px;
    padding:.9rem 1.1rem; margin:.4rem 0; font-size:.85rem; }
.metric-box { background:#111318; border:1px solid #1e2530; border-radius:8px;
    padding:1rem; text-align:center; }
.metric-num { font-family:'IBM Plex Mono',monospace; font-size:1.8rem;
    color:#00ff9d; font-weight:600; }
.metric-lbl { font-size:.75rem; color:#555; margin-top:.2rem; }
.swatch { display:inline-block; width:16px; height:16px; border-radius:3px;
    vertical-align:middle; margin-right:5px; border:1px solid #333; }
.stButton > button { background:#00ff9d !important; color:#000 !important;
    font-family:'IBM Plex Mono',monospace !important; font-weight:600 !important;
    border:none !important; border-radius:6px !important; }
.stButton > button:hover { background:#00cc7a !important; }
.stTextInput > div > div > input { background:#111318 !important;
    border:1px solid #1e2530 !important; color:#e8e8e8 !important; border-radius:6px !important; }
div[data-testid="stSidebar"] { background:#090909 !important; border-right:1px solid #1a1a1a !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_yolo(name="yolov8n.pt"):
    return YOLO(name)

@st.cache_resource
def load_ocr():
    if not OCR_AVAILABLE:
        return None
    return easyocr.Reader(['tr', 'en'], gpu=False)

# ─────────────────────────────────────────────────────────────
# IMAGE ENHANCEMENT
# ─────────────────────────────────────────────────────────────
def enhance(img_rgb: np.ndarray) -> np.ndarray:
    """White-balance + CLAHE + hafif keskinleştirme"""
    # Grey-world white balance
    out = img_rgb.astype(np.float32)
    for c in range(3):
        mean = out[:, :, c].mean()
        if mean > 0:
            out[:, :, c] *= out.mean() / mean
    out = np.clip(out, 0, 255).astype(np.uint8)
    # CLAHE
    lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    # Unsharp mask
    blur = cv2.GaussianBlur(out, (0, 0), 3)
    out  = cv2.addWeighted(out, 1.4, blur, -0.4, 0)
    return np.clip(out, 0, 255).astype(np.uint8)

# ─────────────────────────────────────────────────────────────
# COLOR DETECTION
# ─────────────────────────────────────────────────────────────
COLOR_RANGES = [
    ("kırmızı",  0,  10, 70, 255, 60, 255),
    ("kırmızı", 170, 180, 70, 255, 60, 255),
    ("turuncu",  11,  25, 70, 255, 80, 255),
    ("sarı",     26,  34, 70, 255, 80, 255),
    ("yeşil",    35,  85, 35, 255, 35, 255),
    ("mavi",     86, 130, 35, 255, 35, 255),
    ("lacivert", 100, 130, 70, 255, 20, 110),
    ("mor",      131, 160, 35, 255, 35, 255),
    ("pembe",    161, 175, 35, 255, 60, 255),
    ("beyaz",    0,  180,  0,  30, 210, 255),
    ("siyah",    0,  180,  0, 255,  0,  45),
    ("gri",      0,  180,  0,  40, 46, 209),
]

COLOR_HEX = {
    "kırmızı":"#e53935","turuncu":"#fb8c00","sarı":"#fdd835",
    "yeşil":"#43a047","mavi":"#1e88e5","lacivert":"#1a237e",
    "mor":"#8e24aa","pembe":"#e91e63","beyaz":"#f5f5f5",
    "siyah":"#212121","gri":"#757575"
}

def _pixel_color(h, s, v):
    # Lacivert önce kontrol et
    if 100 <= h <= 130 and s >= 70 and v <= 110:
        return "lacivert"
    for name, h0, h1, s0, s1, v0, v1 in COLOR_RANGES:
        if h0 <= h <= h1 and s0 <= s <= s1 and v0 <= v <= v1:
            return name
    return None

def dominant_colors(img_rgb, top_n=4):
    img_e = enhance(img_rgb)
    small = cv2.resize(img_e, (150, 150))
    hsv   = cv2.cvtColor(small, cv2.COLOR_RGB2HSV).reshape(-1, 3)
    counts = Counter()
    for h, s, v in hsv:
        c = _pixel_color(int(h), int(s), int(v))
        if c:
            counts[c] += 1
    total = sum(counts.values()) or 1
    return [{"color": c, "pct": round(n/total*100)}
            for c, n in counts.most_common(top_n) if round(n/total*100) >= 5]

def object_color(img_rgb, x1, y1, x2, y2):
    """YOLO kutusunun içindeki baskın renk"""
    H, W = img_rgb.shape[:2]
    margin = 0.08
    dx = int((x2-x1)*margin); dy = int((y2-y1)*margin)
    cx1 = max(0, int(x1)+dx); cy1 = max(0, int(y1)+dy)
    cx2 = min(W, int(x2)-dx); cy2 = min(H, int(y2)-dy)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    crop = enhance(img_rgb[cy1:cy2, cx1:cx2])
    if crop.size == 0:
        return None
    hsv  = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV).reshape(-1, 3)
    counts = Counter()
    for h, s, v in hsv:
        c = _pixel_color(int(h), int(s), int(v))
        if c:
            counts[c] += 1
    return counts.most_common(1)[0][0] if counts else None

# ─────────────────────────────────────────────────────────────
# SPATIAL
# ─────────────────────────────────────────────────────────────
def grid_pos(xc, yc, W, H):
    h = "sol" if xc < W/3 else ("orta" if xc < 2*W/3 else "sağ")
    v = "üst" if yc < H/3 else ("orta" if yc < 2*H/3 else "alt")
    return f"{v} {h}"

def relations(objs):
    out = []
    for i, a in enumerate(objs):
        for j, b in enumerate(objs):
            if i >= j:
                continue
            ax, ay = a["xc"], a["yc"]
            bx, by = b["xc"], b["yc"]
            if abs(ax-bx) < 60 and abs(ay-by) < 60:
                out.append(f"{a['object']} ve {b['object']} yan yana")
            elif ay < by - 80:
                out.append(f"{a['object']}, {b['object']}'nin üzerinde")
            elif ax < bx - 80:
                out.append(f"{a['object']}, {b['object']}'nin solunda")
    return out[:5]

# ─────────────────────────────────────────────────────────────
# RULE-BASED ANSWER ENGINE  ← kalp burası
# ─────────────────────────────────────────────────────────────
def rule_answer(question: str, objs: list, colors: list, ocr: list) -> str:
    q  = question.lower().strip().rstrip("?")
    counts = Counter(o["object"] for o in objs)

    # ── Sayma ────────────────────────────────────────────────
    if any(k in q for k in ["kaç", "kaçtane", "kaç tane", "adet", "how many", "count"]):
        if not objs:
            return "Görselde hiç nesne tespit edilemedi."
        # Belirli nesne sorulmuyor mu?
        specific = None
        for obj_name in counts:
            if obj_name in q:
                specific = obj_name
                break
        if specific:
            return f"Görselde {counts[specific]} tane {specific} var."
        total = len(objs)
        detail = ", ".join(f"{v} {k}" for k, v in counts.items())
        return f"Toplamda {total} nesne tespit edildi: {detail}."

    # ── Konum ────────────────────────────────────────────────
    if any(k in q for k in ["nerede", "konumu", "hangi taraf", "where"]):
        target = None
        for obj_name in counts:
            if obj_name in q:
                target = obj_name
                break
        if target:
            matches = [o for o in objs if o["object"] == target]
            if matches:
                pos_list = list({o["position"] for o in matches})
                return f"{target}, görselin {' ve '.join(pos_list)} kısmında."
            return f"Görselde {target} bulunamadı."
        if objs:
            parts = [f"{o['object']} ({o['position']})" for o in objs[:5]]
            return "Tespit edilen nesneler: " + ", ".join(parts) + "."
        return "Görselde nesne tespit edilemedi."

    # ── Renk — NESNE rengi ───────────────────────────────────
    if any(k in q for k in ["renk", "ne renk", "hangi renk", "color", "colour", "renkli"]):
        # Soru belirli bir nesne hakkında mı?
        target = None
        for obj_name in counts:
            if obj_name in q:
                target = obj_name
                break
        if target:
            matches = [o for o in objs if o["object"] == target and o.get("color")]
            if matches:
                c = Counter(o["color"] for o in matches).most_common(1)[0][0]
                return f"{target} {c} renkli."
            return f"{target} tespit edildi ancak rengi belirlenemedi."
        # Genel renk sorusu
        if colors:
            top = colors[0]
            return f"Görseldeki baskın renk {top['color']} (%{top['pct']})."
        return "Renk bilgisi belirlenemedi."

    # ── Metin / OCR ──────────────────────────────────────────
    if any(k in q for k in ["yazı", "yazıyor", "metin", "ne yazı", "text", "okuyor", "yazar"]):
        if ocr:
            return "Görselde şu metinler okundu: " + ", ".join(f'"{t}"' for t in ocr) + "."
        return "Görselde okunabilir metin bulunamadı."

    # ── Varlık sorusu ────────────────────────────────────────
    if any(k in q for k in ["var mı", "mevcut", "is there", "exist", "görünüyor mu", "görüyor"]):
        target = None
        for obj_name in counts:
            if obj_name in q:
                target = obj_name
                break
        if target:
            if target in counts:
                return f"Evet, görselde {counts[target]} tane {target} var."
            return f"Hayır, görselde {target} tespit edilmedi."
        if objs:
            return f"Görselde şunlar var: {', '.join(counts.keys())}."
        return "Görselde nesne tespit edilemedi."

    # ── İlişki sorusu ────────────────────────────────────────
    if any(k in q for k in ["yanında", "üzerinde", "altında", "yakın", "next to", "above", "below"]):
        rels = relations(objs)
        if rels:
            return "Mekansal ilişkiler: " + "; ".join(rels) + "."
        return "Nesneler arasında belirgin mekansal ilişki tespit edilmedi."

    # ── Genel / tanımla ──────────────────────────────────────
    if any(k in q for k in ["ne var", "neler var", "what", "describe", "anlat", "açıkla"]):
        if not objs:
            return "Görselde tanınan nesne bulunamadı."
        parts = []
        for obj_name, cnt in counts.items():
            color_objs = [o for o in objs if o["object"] == obj_name and o.get("color")]
            color_str  = (" (" + color_objs[0]["color"] + ")") if color_objs else ""
            pos_list   = list({o["position"] for o in objs if o["object"] == obj_name})
            parts.append(f"{cnt} {obj_name}{color_str} — {', '.join(pos_list)}")
        answer = "Görselde: " + "; ".join(parts) + "."
        if ocr:
            answer += f' Görsel üzerindeki yazılar: {", ".join(ocr)}.'
        return answer

    # ── Fallback: genel özet ─────────────────────────────────
    if not objs:
        return "Görselde nesne tespit edilemedi."
    parts = [f"{v} {k}" for k, v in counts.items()]
    return "Görselde şunlar tespit edildi: " + ", ".join(parts) + "."

# ─────────────────────────────────────────────────────────────
# MEMORY (JSON)
# ─────────────────────────────────────────────────────────────
MEM_FILE = "memory.json"

def mem_load():
    if not os.path.exists(MEM_FILE):
        return []
    try:
        with open(MEM_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def mem_save(name, objs, colors, ocr, question, answer):
    data = mem_load()
    data.append({
        "image":    name,
        "objects":  objs,
        "colors":   colors,
        "ocr":      ocr,
        "question": question,
        "answer":   answer,
        "ts":       str(datetime.now())
    })
    with open(MEM_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def mem_search(question):
    data  = mem_load()
    words = question.lower().split()
    out   = []
    for item in reversed(data):
        text = " ".join(item.get("objects", []) + item.get("ocr", []) + [item.get("question", "")]).lower()
        if any(w in text for w in words):
            out.append(item)
        if len(out) >= 3:
            break
    return out

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="main-header">VQA Pro</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Visual Question Answering</p>', unsafe_allow_html=True)
    st.markdown("---")

    yolo_model = st.selectbox("YOLO Modeli", ["yolov8n.pt","yolov8s.pt","yolov8m.pt"], index=0)
    yolo_conf  = st.slider("Güven Eşiği", 0.10, 0.90, 0.30, 0.05)
    use_ocr    = st.checkbox("OCR (metin okuma)", value=True)
    use_llava  = st.checkbox("LLaVA ile zenginleştir (opsiyonel)", value=False)

    if use_llava:
        llava_model = st.selectbox("LLaVA", ["llava:latest","llava:13b","llava:34b"])
        st.caption("LLaVA cevabı kural motorunun yanına eklenir.")
    else:
        llava_model = "llava:latest"

    st.markdown("---")
    st.markdown("**Sistem**")
    st.markdown(f"{'🟢' if OCR_AVAILABLE    else '🔴'} EasyOCR")
    st.markdown(f"{'🟢' if OLLAMA_AVAILABLE else '🔴'} Ollama/LLaVA")

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">Visual Question Answering</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">YOLO · Kural Motoru · Renk · OCR · Hafıza</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📁 Dosya Yükle", "📷 Kamera"])
with tab1:
    uploaded = st.file_uploader("Görsel yükle", type=["jpg","jpeg","png"])
    cam      = None
with tab2:
    cam = st.camera_input("Fotoğraf çek")
    if cam:
        uploaded = None

active = cam if cam else uploaded

question   = st.text_input("Sorunuzu yazın", placeholder="Kaç nesne var? / Remote nerede? / Rengi ne?")
run_btn    = st.button("🔍 Analiz Et")

# ─────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────
if run_btn:
    if not active or not question:
        st.warning("Görsel ve soru gerekli.")
        st.stop()

    with st.spinner("Analiz ediliyor..."):

        # Dosya yaz
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tmp.write(active.read())
        img_path = tmp.name

        pil_img  = Image.open(img_path)
        img_rgb  = np.array(pil_img.convert("RGB"))
        is_cam   = cam is not None

        # Kamera iyileştirme
        if is_cam:
            img_rgb = enhance(img_rgb)
            cv2.imwrite(img_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            pil_img = Image.fromarray(img_rgb)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="section-title">Görsel</p>', unsafe_allow_html=True)
            st.image(pil_img, use_container_width=True)

        # ── YOLO ─────────────────────────────────────────────
        yolo   = load_yolo(yolo_model)
        conf_t = max(0.18, yolo_conf - 0.10) if is_cam else yolo_conf
        sizes  = [640, 960] if is_cam else [640]

        seen = set()
        objs = []

        for sz in sizes:
            for result in yolo(img_path, conf=conf_t, imgsz=sz, iou=0.45):
                W, H = result.orig_shape[1], result.orig_shape[0]
                for box in result.boxes:
                    cname = yolo.names[int(box.cls[0])]
                    conf  = float(box.conf[0])
                    x1,y1,x2,y2 = [float(v) for v in box.xyxy[0]]
                    xc = (x1+x2)/2; yc = (y1+y2)/2
                    key = (cname, int(xc//60), int(yc//60))
                    if key in seen:
                        continue
                    seen.add(key)
                    col = object_color(img_rgb, x1, y1, x2, y2)
                    objs.append({
                        "object":   cname,
                        "position": grid_pos(xc, yc, W, H),
                        "confidence": round(conf, 2),
                        "xc": xc, "yc": yc,
                        "color": col
                    })

        # Annotation görseli (son ölçek)
        last_result = yolo(img_path, conf=conf_t, imgsz=sizes[-1])
        with col2:
            st.markdown('<p class="section-title">Nesne Tespiti</p>', unsafe_allow_html=True)
            st.image(last_result[0].plot(), channels="BGR", use_container_width=True)

        counts = Counter(o["object"] for o in objs)
        unique = list(counts.keys())

        # ── Metrikler ─────────────────────────────────────────
        st.markdown('<p class="section-title">Özet</p>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        avg_c = round(sum(o["confidence"] for o in objs)/max(len(objs),1)*100)
        for col, val, lbl in [(m1,len(objs),"Tespit"),(m2,len(unique),"Tür"),(m3,f"%{avg_c}","Güven")]:
            with col:
                st.markdown(f'<div class="metric-box"><div class="metric-num">{val}</div>'
                            f'<div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)

        # ── Nesne etiketleri ──────────────────────────────────
        st.markdown('<p class="section-title">Bulunan Nesneler</p>', unsafe_allow_html=True)
        if unique:
            tags = "".join(f'<span class="obj-tag">{o} ×{counts[o]}</span>' for o in unique)
            st.markdown(f'<div class="result-box">{tags}</div>', unsafe_allow_html=True)
        else:
            st.warning("Nesne bulunamadı.")

        # ── Konum + renk kartları ─────────────────────────────
        st.markdown('<p class="section-title">Konum & Renk</p>', unsafe_allow_html=True)
        for o in objs:
            color_badge = ""
            if o.get("color"):
                hex_c = COLOR_HEX.get(o["color"], "#888")
                color_badge = (f' &nbsp;<span style="background:{hex_c};color:#000;'
                               f'border-radius:3px;padding:1px 7px;font-size:.74rem;">'
                               f'{o["color"]}</span>')
            st.markdown(
                f'<div class="pos-row"><b style="color:#e8e8e8">{o["object"]}</b>'
                f' → {o["position"]} &nbsp; %{int(o["confidence"]*100)}{color_badge}</div>',
                unsafe_allow_html=True)

        # ── OCR ───────────────────────────────────────────────
        ocr_texts = []
        if use_ocr:
            reader = load_ocr()
            if reader:
                with st.spinner("Metin okunuyor..."):
                    ocr_texts = [r.strip() for r in reader.readtext(img_path, detail=0, paragraph=True) if r.strip()]
                st.markdown('<p class="section-title">OCR — Metinler</p>', unsafe_allow_html=True)
                if ocr_texts:
                    tags = "".join(f'<span class="ocr-tag">{t}</span>' for t in ocr_texts)
                    st.markdown(f'<div class="result-box">{tags}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="pos-row">Metin bulunamadı.</div>', unsafe_allow_html=True)

        # ── Renk analizi ──────────────────────────────────────
        colors = dominant_colors(img_rgb)
        st.markdown('<p class="section-title">Baskın Renkler</p>', unsafe_allow_html=True)
        if colors:
            ch = "".join(
                f'<span class="obj-tag">'
                f'<span class="swatch" style="background:{COLOR_HEX.get(c["color"],"#888")}"></span>'
                f'{c["color"]} %{c["pct"]}</span>'
                for c in colors)
            st.markdown(f'<div class="result-box">{ch}</div>', unsafe_allow_html=True)

        # ── Hafıza ────────────────────────────────────────────
        mem_hits = mem_search(question)
        if mem_hits:
            st.markdown('<p class="section-title">Geçmiş Hafıza</p>', unsafe_allow_html=True)
            for m in mem_hits[:3]:
                prev_ans = f'<br><i style="color:#00ff9d;font-size:.8rem">Önceki cevap: {m["answer"]}</i>' if m.get("answer") else ""
                st.markdown(
                    f'<div class="mem-card"><b>{m["image"]}</b>'
                    f' <span style="color:#555;font-size:.78rem">{m.get("ts", m.get("timestamp",""))[:16]}</span><br>'
                    f'Nesneler: {", ".join(m["objects"][:5])}'
                    f'{prev_ans}</div>',
                    unsafe_allow_html=True)

        # ── KURAL MOTORU cevabı ───────────────────────────────
        answer = rule_answer(question, objs, colors, ocr_texts)

        # ── LLaVA (opsiyonel ek) ──────────────────────────────
        llava_note = ""
        if use_llava and OLLAMA_AVAILABLE:
            with st.spinner(f"{llava_model} ek yorum üretiyor..."):
                try:
                    prompt = (
                        "Look at the image. Answer in ONE short Turkish sentence. "
                        "No lists, no bullets. Just the answer.\n"
                        f"Question: {question}"
                    )
                    r = ollama_chat(model=llava_model,
                                    messages=[{"role":"user","content":prompt,"images":[img_path]}])
                    llava_note = r["message"]["content"].strip()
                except Exception:
                    pass

        # ── Cevap kutusu ──────────────────────────────────────
        st.markdown('<p class="section-title">Cevap</p>', unsafe_allow_html=True)
        display = answer
        if llava_note:
            display += f'<br><span style="color:#7eb8ff;font-size:.9rem">LLaVA: {llava_note}</span>'
        st.markdown(f'<div class="ai-answer">{display}</div>', unsafe_allow_html=True)

        # Kaydet
        img_name = active.name if hasattr(active, "name") else "kamera"
        mem_save(img_name, unique, [c["color"] for c in colors], ocr_texts, question, answer)

        try:
            os.unlink(img_path)
        except Exception:
            pass