import streamlit as st
st.set_page_config(page_title="Plant AI | Disease Detection", layout="centered", page_icon="🌿")

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from fpdf import FPDF
import json
import tempfile
import gdown
import os
import streamlit.components.v1 as components

# ----------- CUSTOM CSS -----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d1a0f;
    color: #e8f0e9;
}
.stApp {
    background: radial-gradient(ellipse at top left, #122a15 0%, #0d1a0f 50%, #080f09 100%);
    min-height: 100vh;
}
[data-testid="stSidebar"] { display: none; }
.hero-header {
    text-align: center;
    padding: 3rem 1rem 2rem;
    position: relative;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: 0; left: 50%;
    transform: translateX(-50%);
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(74,163,60,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.5px;
    margin-bottom: 0.3rem;
    line-height: 1.1;
}
.hero-title span { color: #5fd068; }
.hero-subtitle {
    font-size: 1rem;
    color: #7aad7f;
    font-weight: 300;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.hero-desc {
    font-size: 0.95rem;
    color: #6b8f70;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.6;
}
.leaf-divider {
    text-align: center;
    color: #2d5c32;
    font-size: 1.2rem;
    letter-spacing: 8px;
    margin: 1.5rem 0;
}
/* ════════════════════════════════════════
   BACKGROUND ANIMATION KEYFRAMES
   ════════════════════════════════════════ */
/* Aurora sweep */
@keyframes auroraPulse {
    0%,100% { opacity: 0.7; transform: scale(1)    skewX(0deg);  }
    33%     { opacity: 1.0; transform: scale(1.06) skewX(2deg);  }
    66%     { opacity: 0.8; transform: scale(0.96) skewX(-2deg); }
}
@keyframes auroraSwipe {
    0%   { transform: translateX(-15%) skewY(-3deg); opacity:0.6; }
    50%  { transform: translateX( 10%) skewY( 3deg); opacity:1.0; }
    100% { transform: translateX(-15%) skewY(-3deg); opacity:0.6; }
}
/* Botanical corner sway */
@keyframes botanicalSway {
    0%,100% { transform: rotate(-2deg); }
    50%     { transform: rotate( 2deg); }
}
/* Orb pulse + drift */
@keyframes glowPulse {
    0%, 100% { opacity: 0.12; transform: scale(1);    }
    50%       { opacity: 0.30; transform: scale(1.18); }
}
@keyframes driftA {
    0%   { transform: translate(  0px,   0px); }
    25%  { transform: translate( 70px, -55px); }
    50%  { transform: translate(-40px, -90px); }
    75%  { transform: translate(-80px, -30px); }
    100% { transform: translate(  0px,   0px); }
}
@keyframes driftB {
    0%   { transform: translate(  0px,  0px); }
    25%  { transform: translate(-80px, 50px); }
    50%  { transform: translate( 60px, 80px); }
    75%  { transform: translate( 90px,-40px); }
    100% { transform: translate(  0px,  0px); }
}
@keyframes driftC {
    0%   { transform: translate( 0px,   0px); }
    33%  { transform: translate(50px, -70px); }
    66%  { transform: translate(-60px,-40px); }
    100% { transform: translate( 0px,   0px); }
}
@keyframes driftD {
    0%   { transform: translate(0px,  0px); }
    50%  { transform: translate(-50px,60px); }
    100% { transform: translate(0px,  0px); }
}
/* Leaf float up */
@keyframes leafFloat {
    0%   { transform: translateY(105vh) rotate(  0deg) scale(0.3); opacity: 0;   }
    5%   { opacity: 0.8; }
    95%  { opacity: 0.2; }
    100% { transform: translateY( -8vh) rotate(380deg) scale(1.1); opacity: 0;   }
}
@keyframes leafSway {
    0%,100% { transform: translateX(  0px); }
    25%     { transform: translateX( 28px); }
    75%     { transform: translateX(-28px); }
}
/* Grid shimmer */
@keyframes gridShimmer {
    0%,100% { opacity: 0.03; }
    50%     { opacity: 0.07; }
}
/* Ripple rings */
@keyframes ripple {
    0%   { transform: scale(0.4); opacity: 0.5; }
    100% { transform: scale(2.8); opacity: 0;   }
}
/* Shooting star */
@keyframes shoot {
    0%   { transform: translateX(-100px) translateY( 0px) rotate(-25deg); opacity: 0; }
    5%   { opacity: 0.8; }
    60%  { opacity: 0.4; }
    100% { transform: translateX(110vw)  translateY(40px) rotate(-25deg); opacity: 0; }
}
/* Twinkling dots */
@keyframes twinkle {
    0%,100% { opacity: 0.08; transform: scale(1);   }
    50%     { opacity: 0.45; transform: scale(1.6); }
}

/* ════════════════════════════════════════
   RADIO GROUP — FORCE CENTRE
   ════════════════════════════════════════ */
/* Nuke any Streamlit column width constraints above radio */
[data-testid="stHorizontalBlock"] { justify-content: center !important; }

div[role="radiogroup"] {
    display:         flex         !important;
    flex-direction:  row          !important;
    flex-wrap:       wrap         !important;
    justify-content: center       !important;
    align-items:     center       !important;
    gap:             14px         !important;
    margin:          0.5rem auto 2rem !important;
    width:           100%         !important;
    padding:         0            !important;
}
div[role="radiogroup"] > div,
div[role="radiogroup"] > label { flex: 0 0 auto !important; }
/* Hide the native radio bullet */
div[role="radiogroup"] [data-baseweb="radio"] > div:first-child,
div[role="radiogroup"] input[type="radio"] { display: none !important; }
div[role="radiogroup"] label {
    background:    rgba(255,255,255,0.05)       !important;
    border:        1px solid rgba(95,208,104,0.28) !important;
    border-radius: 50px                          !important;
    padding:       13px 32px                     !important;
    cursor:        pointer                       !important;
    font-size:     0.93rem                       !important;
    font-weight:   500                           !important;
    color:         #a8cead                       !important;
    transition:    all 0.28s ease                !important;
    white-space:   nowrap                        !important;
    letter-spacing:0.03em                        !important;
    box-shadow:    0 2px 12px rgba(0,0,0,0.3)   !important;
}
div[role="radiogroup"] label:hover {
    background:  rgba(95,208,104,0.14)          !important;
    border-color:rgba(95,208,104,0.7)           !important;
    color:       #5fd068                        !important;
    box-shadow:  0 0 28px rgba(95,208,104,0.25),
                 0 4px 16px rgba(0,0,0,0.4)     !important;
    transform:   translateY(-3px)               !important;
}
/* Centre the entire Streamlit block that wraps the radio */
[data-testid="stRadio"] {
    display:         flex         !important;
    justify-content: center       !important;
    width:           100%         !important;
}
[data-testid="stRadio"] > div {
    display:         flex         !important;
    justify-content: center       !important;
    width:           100%         !important;
}
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 2px dashed rgba(95,208,104,0.25) !important;
    border-radius: 20px !important;
    padding: 2rem !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(95,208,104,0.5) !important;
    background: rgba(95,208,104,0.04) !important;
}
[data-testid="stFileUploader"] * { color: #7aad7f !important; }
[data-testid="stImage"] img {
    border-radius: 16px !important;
    border: 1px solid rgba(95,208,104,0.15) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4) !important;
}
.result-card {
    background: linear-gradient(135deg, rgba(18,42,21,0.9) 0%, rgba(13,26,15,0.9) 100%);
    border: 1px solid rgba(95,208,104,0.2);
    border-radius: 20px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
    position: relative;
    overflow: hidden;
}
.result-label {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    color: #ffffff;
    font-weight: 600;
    margin-bottom: 0.3rem;
}
.result-confidence {
    font-size: 0.85rem;
    color: #5fd068;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.confidence-bar-bg {
    background: rgba(255,255,255,0.08);
    border-radius: 50px;
    height: 6px;
    margin: 0.8rem 0;
    overflow: hidden;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 50px;
    background: linear-gradient(90deg, #3a9e45, #5fd068);
}
.info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 1rem;
}
.info-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(95,208,104,0.12);
    border-radius: 14px;
    padding: 1.2rem;
}
.info-card-icon { font-size: 1.4rem; margin-bottom: 0.5rem; }
.info-card-title {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #5fd068;
    font-weight: 500;
    margin-bottom: 0.4rem;
}
.info-card-text { font-size: 0.88rem; color: #c5d8c7; line-height: 1.5; }
.badge-healthy {
    display: inline-block;
    background: rgba(95,208,104,0.15);
    border: 1px solid #5fd068;
    color: #5fd068;
    border-radius: 50px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.badge-disease {
    display: inline-block;
    background: rgba(255,100,80,0.12);
    border: 1px solid rgba(255,100,80,0.5);
    color: #ff8a7a;
    border-radius: 50px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(135deg, #2d7a35, #3a9e45) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.04em !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 16px rgba(58,158,69,0.3) !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(58,158,69,0.45) !important;
}
.stats-row {
    display: flex;
    justify-content: center;
    gap: 1.5rem;
    flex-wrap: wrap;
    margin: 2rem auto 0;
    max-width: 600px;
}
.stat-pill {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(95,208,104,0.18);
    border-radius: 50px;
    padding: 0.55rem 1.4rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.82rem;
    color: #7aad7f;
    white-space: nowrap;
}
.stat-pill strong { color: #5fd068; font-size: 0.9rem; }
.leaf-strip {
    text-align: center;
    font-size: 0.9rem;
    letter-spacing: 14px;
    color: rgba(95,208,104,0.15);
    margin: 1.5rem 0 0.5rem;
    user-select: none;
}
.section-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #4a8c50;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::before, .section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(95,208,104,0.15);
}
.footer {
    text-align: center;
    padding: 3rem 1rem 2rem;
    color: #2d5c32;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)

# ----------- MODEL LOAD -----------
MODEL_PATH = "plant_disease_model.h5"
FILE_ID = "1MKW4o4Ux-8uMcO7QukDPwwa2GgLW01dq"

if not os.path.exists(MODEL_PATH):
    st.warning("⬇️ Downloading model from Google Drive...")
    try:
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
        if os.path.getsize(MODEL_PATH) < 1000000:
            st.error("❌ Downloaded file is corrupted!")
            os.remove(MODEL_PATH)
            st.stop()
        st.success("✅ Model downloaded successfully!")
    except Exception as e:
        st.error("❌ Auto-download failed!")
        st.write(e)
        st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error("❌ Model loading failed (file may be corrupted)")
    st.write(e)
    st.stop()

IMG_SIZE = 224

# ----------- CLASS NAMES -----------
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
class_names = list(class_indices.keys())

# ----------- DISEASE INFO -----------
disease_info = {
    "Pepper__bell___Bacterial_spot": {
        "treatment": "Use disease-free seeds and apply bactericides.",
        "prevention": "Avoid overhead watering and maintain plant hygiene."
    },
    "Pepper__bell___healthy": {
        "treatment": "No treatment needed.",
        "prevention": "Maintain proper care."
    },
    "Potato___Early_blight": {
        "treatment": "Apply fungicides and remove infected leaves.",
        "prevention": "Rotate crops and avoid moisture on leaves."
    },
    "Potato___healthy": {
        "treatment": "No treatment needed.",
        "prevention": "Proper watering and nutrients."
    },
    "Potato___Late_blight": {
        "treatment": "Use fungicides and remove infected plants.",
        "prevention": "Avoid high humidity and overcrowding."
    },
    "Tomato_Target_Spot": {
        "treatment": "Apply fungicides and remove infected parts.",
        "prevention": "Ensure proper airflow."
    },
    "Tomato_Tomato_mosaic_virus": {
        "treatment": "Remove infected plants immediately.",
        "prevention": "Use resistant varieties and sanitize tools."
    },
    "Tomato__Tomato_YellowLeafCurl_Virus": {
        "treatment": "Control whiteflies and remove infected plants.",
        "prevention": "Use insect-proof nets."
    },
    "Tomato_Bacterial_spot": {
        "treatment": "Apply copper-based bactericides.",
        "prevention": "Avoid wet leaves."
    },
    "Tomato_Early_blight": {
        "treatment": "Use fungicides and remove infected leaves.",
        "prevention": "Avoid overhead watering."
    },
    "Tomato_healthy": {
        "treatment": "No treatment needed.",
        "prevention": "Maintain proper care."
    },
    "Tomato_Late_blight": {
        "treatment": "Apply fungicides and remove infected plants.",
        "prevention": "Ensure airflow and avoid excess moisture."
    },
    "Tomato_Leaf_Mold": {
        "treatment": "Use fungicides.",
        "prevention": "Reduce humidity and improve ventilation."
    },
    "Tomato_Septoria_leaf_spot": {
        "treatment": "Remove infected leaves and apply fungicide.",
        "prevention": "Avoid water splash on leaves."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "treatment": "Use miticides or neem oil.",
        "prevention": "Maintain humidity and wash leaves."
    }
}

# ----------- LEAF CHECK -----------
def is_leaf(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_pixels = np.sum(mask > 0)
    total_pixels = frame.shape[0] * frame.shape[1]
    return (green_pixels / total_pixels) > 0.08

# ----------- PREDICT -----------
def predict_image(image):
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    if not is_leaf(frame):
        return "Not a plant leaf", 0, {
            "treatment": "Please upload a clear leaf image.",
            "prevention": "Ensure the leaf is the main subject with minimal background noise."
        }
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = np.reshape(img, [1, IMG_SIZE, IMG_SIZE, 3])
    prediction = model.predict(img, verbose=0)
    index = np.argmax(prediction)
    confidence = prediction[0][index] * 100
    label = class_names[index]
    info = disease_info.get(label, {"treatment": "No data available", "prevention": "No data available"})
    return label, confidence, info

# ----------- VIDEO PROCESSOR -----------
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.latest_frame = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.latest_frame = img
        if not is_leaf(img):
            cv2.putText(img, "Not a leaf", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return img
        resized = cv2.resize(img, (224, 224))
        normalized = resized / 255.0
        input_img = normalized.reshape(1, 224, 224, 3)
        pred = model.predict(input_img, verbose=0)
        index = np.argmax(pred)
        confidence = pred[0][index] * 100
        label = class_names[index]
        cv2.putText(img, f"{label} ({confidence:.1f}%)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 220, 90), 2)
        return img

# ----------- PDF -----------
def build_pdf_page(pdf, image, label, confidence, treatment, prevention, fname=""):
    from datetime import datetime
    pdf.add_page()
    margin = 20
    page_w = 210

    pdf.set_font("Times", "B", 20)
    pdf.set_text_color(15, 80, 20)
    pdf.cell(0, 12, "Plant AI  -  Disease Detection", ln=True, align="C")
    pdf.set_font("Times", "I", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 6, "AI-Powered Plant Diagnostics System", ln=True, align="C")

    pdf.set_draw_color(60, 160, 70)
    pdf.set_line_width(0.6)
    pdf.line(margin, pdf.get_y() + 2, page_w - margin, pdf.get_y() + 2)
    pdf.ln(6)

    pdf.set_font("Times", "", 9)
    pdf.set_text_color(120, 120, 120)
    now = datetime.now().strftime("%d %B %Y  %H:%M")
    pdf.cell(0, 5, f"Generated: {now}", ln=True, align="R")
    if fname:
        clean_fname = str(fname).encode("latin-1", "ignore").decode("latin-1")
        pdf.cell(0, 5, f"File: {clean_fname}", ln=True, align="R")
    pdf.ln(4)

    pdf.set_font("Times", "B", 11)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 7, "Uploaded Image:", ln=True)
    pdf.ln(1)

    img_w = 100
    img_h = 75
    img_x = (page_w - img_w) / 2
    img_y = pdf.get_y()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name, format="JPEG")
        img_path = tmp.name
    pdf.image(img_path, x=img_x, y=img_y, w=img_w, h=img_h)
    pdf.set_y(img_y + img_h + 6)

    pdf.set_draw_color(200, 220, 200)
    pdf.set_line_width(0.3)
    pdf.line(margin, pdf.get_y(), page_w - margin, pdf.get_y())
    pdf.ln(5)

    pdf.set_font("Times", "B", 13)
    pdf.set_text_color(15, 80, 20)
    pdf.cell(0, 8, "Analysis", ln=True)
    pdf.ln(1)

    clean_label = label.encode("latin-1", "ignore").decode("latin-1")
    display_label = clean_label.replace("_", " ").replace("  ", " ")
    is_healthy = "healthy" in clean_label.lower()
    is_not_leaf = "not a plant" in clean_label.lower()

    pdf.set_font("Times", "B", 11)
    pdf.set_text_color(30, 30, 30)
    if is_not_leaf:
        status_text, status_val = "Status:", "Not a plant leaf"
    elif is_healthy:
        status_text, status_val = "Status:", "Healthy Plant"
    else:
        status_text, status_val = "Disease Detected:", display_label

    pdf.cell(45, 7, status_text)
    pdf.set_font("Times", "", 11)
    if is_healthy:
        pdf.set_text_color(0, 120, 20)
    else:
        pdf.set_text_color(180, 50, 30)
    pdf.cell(0, 7, status_val, ln=True)

    pdf.set_text_color(30, 30, 30)
    pdf.set_font("Times", "B", 11)
    pdf.cell(45, 7, "Confidence:")
    pdf.set_font("Times", "", 11)
    pdf.cell(0, 7, f"{confidence:.2f}%", ln=True)
    pdf.ln(3)

    pdf.set_draw_color(200, 220, 200)
    pdf.set_line_width(0.3)
    pdf.line(margin, pdf.get_y(), page_w - margin, pdf.get_y())
    pdf.ln(4)

    pdf.set_font("Times", "B", 11)
    pdf.set_text_color(15, 80, 20)
    pdf.cell(0, 7, "Treatment Recommendation:", ln=True)
    pdf.set_font("Times", "", 11)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(0, 6, treatment.encode("latin-1", "ignore").decode("latin-1"))
    pdf.ln(3)

    pdf.set_font("Times", "B", 11)
    pdf.set_text_color(15, 80, 20)
    pdf.cell(0, 7, "Prevention Tips:", ln=True)
    pdf.set_font("Times", "", 11)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(0, 6, prevention.encode("latin-1", "ignore").decode("latin-1"))
    pdf.ln(4)

    pdf.set_draw_color(200, 220, 200)
    pdf.set_line_width(0.3)
    pdf.line(margin, pdf.get_y(), page_w - margin, pdf.get_y())
    pdf.ln(3)
    pdf.set_font("Times", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.multi_cell(0, 5, "This report is generated by an AI model and is intended as a preliminary diagnostic aid only. "
                         "Please consult an agronomist or plant pathologist for confirmed diagnosis and treatment.")


def generate_pdf(image, label, confidence, treatment, prevention, fname=""):
    pdf = FPDF()
    pdf.set_margins(20, 20, 20)
    build_pdf_page(pdf, image, label, confidence, treatment, prevention, fname)
    return pdf.output(dest='S').encode('latin-1')


# ----------- RENDER RESULT -----------
def render_result(image, label, confidence, info):
    col1, col2 = st.columns([1, 1.6], gap="medium")
    with col1:
        st.image(image, use_container_width=True)
    with col2:
        is_healthy = "healthy" in label.lower()
        is_not_leaf = "not a plant" in label.lower()
        if is_not_leaf:
            badge = '<span class="badge-disease">⚠ Not a Leaf</span>'
        elif is_healthy:
            badge = '<span class="badge-healthy">✓ Healthy</span>'
        else:
            badge = '<span class="badge-disease">⚠ Disease Detected</span>'

        clean_label = label.replace("_", " ").replace("  ", " ")
        conf_bar = min(confidence, 100)

        st.markdown(f"""
        <div class="result-card">
            {badge}
            <div class="result-label">{clean_label}</div>
            <div class="result-confidence">Confidence: {confidence:.1f}%</div>
            <div class="confidence-bar-bg">
                <div class="confidence-bar-fill" style="width:{conf_bar}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-grid">
        <div class="info-card">
            <div class="info-card-icon">💊</div>
            <div class="info-card-title">Treatment</div>
            <div class="info-card-text">{info['treatment']}</div>
        </div>
        <div class="info-card">
            <div class="info-card-icon">🛡️</div>
            <div class="info-card-title">Prevention</div>
            <div class="info-card-text">{info['prevention']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ----------- ANIMATED BACKGROUND -----------
# Static CSS layers via markdown (no JS needed)
st.markdown("""
<div id="staticBG" style="position:fixed;inset:0;pointer-events:none;z-index:0;overflow:hidden;">

  <div style="position:absolute;inset:0;
    background:radial-gradient(ellipse 80% 60% at 15% 10%,rgba(30,110,50,0.28) 0%,transparent 65%),
               radial-gradient(ellipse 70% 55% at 85% 20%,rgba(10,70,25,0.22) 0%,transparent 60%),
               radial-gradient(ellipse 90% 50% at 50% 100%,rgba(20,90,35,0.18) 0%,transparent 60%);
    animation:auroraPulse 12s ease-in-out infinite;">
  </div>
  <div style="position:absolute;top:-40%;left:-20%;width:140%;height:80%;
    background:linear-gradient(165deg,transparent 0%,rgba(58,158,69,0.07) 35%,rgba(95,208,104,0.11) 48%,transparent 72%);
    filter:blur(32px);animation:auroraSwipe 20s ease-in-out infinite;"></div>

  <div style="position:absolute;width:640px;height:640px;border-radius:50%;background:radial-gradient(circle,rgba(58,158,69,0.22),transparent 62%);filter:blur(90px);top:-250px;left:-220px;animation:glowPulse 6s ease-in-out infinite,driftA 22s ease-in-out infinite;"></div>
  <div style="position:absolute;width:500px;height:500px;border-radius:50%;background:radial-gradient(circle,rgba(20,100,38,0.20),transparent 62%);filter:blur(80px);top:28%;right:-180px;animation:glowPulse 8s ease-in-out infinite 2s,driftB 26s ease-in-out infinite 5s;"></div>
  <div style="position:absolute;width:420px;height:420px;border-radius:50%;background:radial-gradient(circle,rgba(95,208,104,0.16),transparent 62%);filter:blur(75px);bottom:-150px;left:25%;animation:glowPulse 7s ease-in-out infinite 4s,driftC 20s ease-in-out infinite 8s;"></div>
  <div style="position:absolute;width:320px;height:320px;border-radius:50%;background:radial-gradient(circle,rgba(30,145,58,0.18),transparent 62%);filter:blur(70px);top:52%;left:3%;animation:glowPulse 9s ease-in-out infinite 1s,driftD 28s ease-in-out infinite 3s;"></div>
  <div style="position:absolute;width:280px;height:280px;border-radius:50%;background:radial-gradient(circle,rgba(70,185,82,0.13),transparent 62%);filter:blur(65px);top:12%;left:42%;animation:glowPulse 5s ease-in-out infinite 3s,driftA 18s ease-in-out infinite reverse;"></div>

  <svg style="position:absolute;inset:0;width:100%;height:100%;" xmlns="http://www.w3.org/2000/svg">
    <defs><pattern id="hexgrid" x="0" y="0" width="56" height="64" patternUnits="userSpaceOnUse">
      <polygon points="28,2 52,16 52,48 28,62 4,48 4,16" fill="none" stroke="rgba(95,208,104,0.055)" stroke-width="0.8"/>
    </pattern></defs>
    <rect width="100%" height="100%" fill="url(#hexgrid)" style="animation:gridShimmer 8s ease-in-out infinite;"/>
  </svg>

  <div style="position:absolute;width:4px;height:4px;border-radius:50%;background:#5fd068;top:8%;left:18%;animation:twinkle 2.8s ease-in-out infinite;"></div>
  <div style="position:absolute;width:3px;height:3px;border-radius:50%;background:#5fd068;top:18%;left:74%;animation:twinkle 4.2s ease-in-out infinite 1s;"></div>
  <div style="position:absolute;width:5px;height:5px;border-radius:50%;background:#3a9e45;top:30%;left:91%;animation:twinkle 3.5s ease-in-out infinite 0.5s;"></div>
  <div style="position:absolute;width:4px;height:4px;border-radius:50%;background:#3a9e45;top:60%;left:52%;animation:twinkle 5.0s ease-in-out infinite 1.5s;"></div>
  <div style="position:absolute;width:3px;height:3px;border-radius:50%;background:#5fd068;top:82%;left:80%;animation:twinkle 2.5s ease-in-out infinite 4s;"></div>

  <div style="position:absolute;width:180px;height:1.5px;background:linear-gradient(90deg,transparent,rgba(95,208,104,0.85),transparent);top:12%;left:0;animation:shoot 7s linear infinite;"></div>
  <div style="position:absolute;width:110px;height:1px;background:linear-gradient(90deg,transparent,rgba(95,208,104,0.65),transparent);top:55%;left:0;animation:shoot 11s linear infinite 3s;"></div>
  <div style="position:absolute;width:220px;height:1.5px;background:linear-gradient(90deg,transparent,rgba(58,158,69,0.75),transparent);top:78%;left:0;animation:shoot 9s linear infinite 6s;"></div>

  <div style="position:absolute;width:200px;height:200px;border-radius:50%;border:1px solid rgba(95,208,104,0.15);top:20%;left:5%;animation:ripple 5s ease-out infinite;"></div>
  <div style="position:absolute;width:200px;height:200px;border-radius:50%;border:1px solid rgba(95,208,104,0.09);top:20%;left:5%;animation:ripple 5s ease-out infinite 1.7s;"></div>
  <div style="position:absolute;width:160px;height:160px;border-radius:50%;border:1px solid rgba(95,208,104,0.12);top:58%;right:6%;animation:ripple 7s ease-out infinite 1s;"></div>
  <div style="position:absolute;width:160px;height:160px;border-radius:50%;border:1px solid rgba(95,208,104,0.07);top:58%;right:6%;animation:ripple 7s ease-out infinite 3.5s;"></div>

  <svg style="position:absolute;inset:0;width:100%;height:100%;opacity:0.06;" xmlns="http://www.w3.org/2000/svg">
    <g style="animation:botanicalSway 9s ease-in-out infinite;transform-origin:60px 220px;">
      <ellipse cx="60" cy="200" rx="45" ry="130" fill="rgba(58,158,69,0.5)" stroke="rgba(95,208,104,0.8)" stroke-width="1" transform="rotate(-28,60,200)"/>
      <line x1="60" y1="90" x2="60" y2="300" stroke="rgba(95,208,104,0.7)" stroke-width="1.2"/>
      <line x1="60" y1="135" x2="28" y2="158" stroke="rgba(95,208,104,0.5)" stroke-width="0.8"/>
      <line x1="60" y1="168" x2="22" y2="190" stroke="rgba(95,208,104,0.4)" stroke-width="0.8"/>
      <line x1="60" y1="135" x2="92" y2="158" stroke="rgba(95,208,104,0.5)" stroke-width="0.8"/>
    </g>
    <g style="animation:botanicalSway 10s ease-in-out infinite 4s;" transform="translate(1440,0) scale(-1,1)">
      <ellipse cx="60" cy="180" rx="38" ry="115" fill="rgba(58,158,69,0.5)" stroke="rgba(95,208,104,0.7)" stroke-width="1" transform="rotate(-22,60,180)"/>
      <line x1="60" y1="80" x2="60" y2="270" stroke="rgba(95,208,104,0.6)" stroke-width="1.2"/>
      <line x1="60" y1="120" x2="32" y2="142" stroke="rgba(95,208,104,0.5)" stroke-width="0.8"/>
      <line x1="60" y1="150" x2="88" y2="170" stroke="rgba(95,208,104,0.5)" stroke-width="0.8"/>
    </g>
    <g style="animation:botanicalSway 11s ease-in-out infinite 1s;" transform="translate(0,900) scale(1,-1)">
      <ellipse cx="55" cy="145" rx="42" ry="120" fill="rgba(58,158,69,0.4)" stroke="rgba(95,208,104,0.6)" stroke-width="1" transform="rotate(-20,55,145)"/>
      <line x1="55" y1="40" x2="55" y2="240" stroke="rgba(95,208,104,0.5)" stroke-width="1.2"/>
    </g>
    <g style="animation:botanicalSway 13s ease-in-out infinite 3s;" transform="translate(1440,900) scale(-1,-1)">
      <ellipse cx="60" cy="140" rx="36" ry="105" fill="rgba(40,140,58,0.4)" stroke="rgba(95,208,104,0.6)" stroke-width="1" transform="rotate(-18,60,140)"/>
      <line x1="60" y1="45" x2="60" y2="225" stroke="rgba(95,208,104,0.5)" stroke-width="1.2"/>
    </g>
  </svg>
</div>
""", unsafe_allow_html=True)

# JS canvas leaf animation via components (bypasses Streamlit script stripping)
components.html("""
<!DOCTYPE html>
<html>
<head>
<style>*{margin:0;padding:0;}html,body{background:transparent;}</style>
</head>
<body>
<script>
/* ── Inject canvas into PARENT document (same-origin Streamlit) ── */
const pDoc = window.parent.document;

// Remove old canvas if re-running
const old = pDoc.getElementById('leafCanvasGlobal');
if(old) old.remove();

const canvas = pDoc.createElement('canvas');
canvas.id = 'leafCanvasGlobal';
canvas.style.cssText = [
  'position:fixed','inset:0','width:100vw','height:100vh',
  'pointer-events:none','z-index:9999','top:0','left:0',
].join(';');
pDoc.body.appendChild(canvas);

const ctx = canvas.getContext('2d');

function resize(){
  canvas.width  = window.parent.innerWidth  || window.screen.width;
  canvas.height = window.parent.innerHeight || window.screen.height;
}
resize();
window.parent.addEventListener('resize', resize);

/* ─── Leaf colours ─── */
const COLORS = [
  [95,208,104],
  [58,158,69],
  [40,140,55],
  [130,220,100],
  [75,180,82],
  [20,110,40],
  [160,230,120],
];

/* ─── Draw a realistic leaf shape ─── */
function drawLeaf(ctx, size, rgb, alpha){
  const [r,g,b] = rgb;
  ctx.save();
  ctx.globalAlpha = alpha;

  // leaf body
  ctx.beginPath();
  ctx.moveTo(0, -size);
  ctx.bezierCurveTo( size*0.95, -size*0.3,  size*0.95,  size*0.3, 0,  size);
  ctx.bezierCurveTo(-size*0.95,  size*0.3, -size*0.95, -size*0.3, 0, -size);
  ctx.fillStyle   = `rgba(${r},${g},${b},${alpha})`;
  ctx.strokeStyle = `rgba(${Math.min(r+30,255)},${Math.min(g+30,255)},${b},${alpha*0.5})`;
  ctx.lineWidth   = 0.6;
  ctx.fill();
  ctx.stroke();

  // midrib
  ctx.beginPath();
  ctx.moveTo(0,-size*0.9);
  ctx.lineTo(0, size*0.9);
  ctx.strokeStyle = `rgba(${Math.min(r+40,255)},${Math.min(g+40,255)},${b},${alpha*0.55})`;
  ctx.lineWidth = 0.5;
  ctx.stroke();

  // veins
  for(let i=1;i<=4;i++){
    const t  = i/5;
    const y  = -size*0.75 + size*1.5*t;
    const vx = size * 0.65 * Math.sin(t*Math.PI);
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.quadraticCurveTo( vx*0.5, y-size*0.05,  vx, y-size*0.12);
    ctx.moveTo(0, y);
    ctx.quadraticCurveTo(-vx*0.5, y-size*0.05, -vx, y-size*0.12);
    ctx.strokeStyle = `rgba(${Math.min(r+50,255)},${Math.min(g+50,255)},${b},${alpha*0.30})`;
    ctx.lineWidth = 0.3;
    ctx.stroke();
  }
  ctx.restore();
}

/* ─── Ripple ─── */
class Ripple {
  constructor(x, y, size){
    this.x     = x;
    this.y     = y;
    this.size  = size;
    this.rings = [
      {r: size*0.2, maxR: size*5.5, alpha:0.75, speed: size*0.22},
      {r: size*0.2, maxR: size*4.5, alpha:0.55, speed: size*0.18},
      {r: size*0.2, maxR: size*3.5, alpha:0.38, speed: size*0.14},
    ];
    // splash dots
    this.dots = Array.from({length:8}, (_,i)=>{
      const ang = (i/8)*Math.PI*2;
      return {
        ang,
        dist: 0,
        maxDist: size*(1.8+Math.random()),
        alpha: 0.85,
        speed: size*0.08 + Math.random()*size*0.06,
      };
    });
    this.done = false;
  }
  update(){
    let alive = false;
    for(const ring of this.rings){
      ring.r     += ring.speed;
      ring.alpha -= 0.013;
      if(ring.alpha > 0) alive = true;
    }
    for(const d of this.dots){
      if(d.dist < d.maxDist){
        d.dist  += d.speed;
        d.alpha -= 0.022;
        alive = true;
      }
    }
    if(!alive) this.done = true;
  }
  draw(){
    // elliptical rings (flat perspective)
    for(const ring of this.rings){
      if(ring.alpha <= 0) continue;
      ctx.beginPath();
      ctx.ellipse(this.x, this.y, ring.r, ring.r*0.28, 0, 0, Math.PI*2);
      ctx.strokeStyle = `rgba(95,208,104,${ring.alpha.toFixed(3)})`;
      ctx.lineWidth   = 1.4;
      ctx.stroke();
    }
    // splash particles
    for(const d of this.dots){
      if(d.alpha <= 0 || d.dist >= d.maxDist) continue;
      const px = this.x + Math.cos(d.ang)*d.dist;
      const py = this.y + Math.sin(d.ang)*d.dist*0.28;
      ctx.beginPath();
      ctx.arc(px, py, 2.2, 0, Math.PI*2);
      ctx.fillStyle = `rgba(95,208,104,${d.alpha.toFixed(3)})`;
      ctx.fill();
    }
  }
}

/* ─── Leaf ─── */
class Leaf {
  constructor(randomY){
    this.rgb     = COLORS[Math.floor(Math.random()*COLORS.length)];
    this.size    = 7 + Math.random()*12;
    this.x       = Math.random()*canvas.width;
    this.y       = randomY ? Math.random()*canvas.height : -this.size*2;
    this.rot     = Math.random()*Math.PI*2;
    this.rotSpd  = (Math.random()-0.5)*0.030;
    this.vy      = 0.45 + Math.random()*1.1;
    this.vx      = (Math.random()-0.5)*0.55;
    this.swayAmp = 22 + Math.random()*30;
    this.swayFq  = 0.007 + Math.random()*0.011;
    this.swayOff = Math.random()*Math.PI*2;
    this.alpha   = 0.18 + Math.random()*0.55;
    this.t       = Math.random()*1000;
    this.landed  = false;
    this.landRot = 0;
    this.fadeAlpha = 0;
    this.fadeSpd = 0.003 + Math.random()*0.004;
  }
  reset(){
    this.x       = Math.random()*canvas.width;
    this.y       = -this.size*2;
    this.rot     = Math.random()*Math.PI*2;
    this.rotSpd  = (Math.random()-0.5)*0.030;
    this.vy      = 0.45 + Math.random()*1.1;
    this.vx      = (Math.random()-0.5)*0.55;
    this.swayAmp = 22 + Math.random()*30;
    this.swayFq  = 0.007 + Math.random()*0.011;
    this.swayOff = Math.random()*Math.PI*2;
    this.alpha   = 0.18 + Math.random()*0.55;
    this.t       = 0;
    this.landed  = false;
  }
  update(){
    if(this.landed){
      this.fadeAlpha -= this.fadeSpd;
      if(this.fadeAlpha <= 0) this.reset();
      return;
    }
    this.t  += 1;
    this.x  += this.vx + Math.sin(this.t*this.swayFq + this.swayOff)*0.8;
    this.y  += this.vy;
    this.rot += this.rotSpd + Math.sin(this.t*this.swayFq)*0.008;

    if(this.y > canvas.height + this.size){
      this.landed    = true;
      this.fadeAlpha = this.alpha;
      this.landRot   = this.rot;
      this.y         = canvas.height - this.size*0.3;
      ripples.push(new Ripple(this.x, this.y, this.size));
    }
  }
  draw(){
    ctx.save();
    ctx.translate(this.x, this.y);
    if(!this.landed){
      const tilt = Math.sin(this.t*this.swayFq + this.swayOff)*0.35;
      ctx.rotate(this.rot + tilt);
      drawLeaf(ctx, this.size, this.rgb, this.alpha);
    } else {
      // lies flat on ground
      ctx.rotate(this.landRot);
      ctx.scale(1, 0.28);
      drawLeaf(ctx, this.size, this.rgb, this.fadeAlpha);
    }
    ctx.restore();
  }
}

/* ─── Init ─── */
const ripples = [];
const LEAF_COUNT = 42;
const leaves = Array.from({length:LEAF_COUNT}, () => new Leaf(true));

/* ─── Loop ─── */
function loop(){
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // ripples
  for(let i=ripples.length-1; i>=0; i--){
    ripples[i].update();
    ripples[i].draw();
    if(ripples[i].done) ripples.splice(i,1);
  }

  // leaves
  for(const leaf of leaves){
    leaf.update();
    leaf.draw();
  }

  requestAnimationFrame(loop);
}
loop();
</script>
</body>
</html>
""", height=1, scrolling=False)
# ----------- HEADER -----------
st.markdown("""
<div class="hero-header">
    <div class="hero-subtitle">AI-Powered Diagnostics</div>
    <div class="hero-title">Plant <span>Disease</span><br>Detection</div>
    <div class="leaf-divider">✦ ✦ ✦</div>
    <div class="hero-desc">Upload a photo of your plant leaf for instant AI diagnosis, treatment recommendations, and prevention tips.</div>
</div>
<div class="stats-row">
    <div class="stat-pill">⚡ <strong>~2s</strong> diagnosis time</div>
    <div class="stat-pill">🎯 <strong>CNN</strong> deep learning</div>
    <div class="stat-pill">📄 <strong>PDF</strong> reports</div>
</div>
<div class="leaf-strip">🍃 🌱 🍃 🌱 🍃 🌱 🍃 🌱 🍃</div>
""", unsafe_allow_html=True)

# ----------- MODE SELECTOR -----------
st.markdown('<div class="section-label">Choose Input Mode</div>', unsafe_allow_html=True)
_c1, _c2, _c3 = st.columns([1, 2, 1])
with _c2:
    option = st.radio("Choose mode", ["📤  Upload Image", "📷  Camera", "🎥  Live Detection"],
                      horizontal=True, label_visibility="collapsed")
st.write("")

# ----------- UPLOAD -----------
if option == "📤  Upload Image":
    files = st.file_uploader("Upload images", type=["jpg", "png", "jpeg"],
                             label_visibility="collapsed", accept_multiple_files=True)
    if files:
        st.markdown(
            f'<div class="section-label">Analysis Results — {len(files)} image{"s" if len(files) > 1 else ""}</div>',
            unsafe_allow_html=True
        )

        all_results = []
        for i, file in enumerate(files):
            image = Image.open(file)
            label, confidence, info = predict_image(image)
            fname = file.name if file.name else f"image_{i+1}.jpg"
            all_results.append((image, label, confidence, info, fname))

            if len(files) > 1:
                st.markdown(
                    f'<div class="section-label" style="font-size:0.65rem;opacity:0.7;">Image {i+1} · {fname}</div>',
                    unsafe_allow_html=True
                )
            render_result(image, label, confidence, info)
            if i < len(files) - 1:
                st.divider()

        st.write("")
        if len(all_results) == 1:
            image, label, confidence, info, fname = all_results[0]
            pdf = generate_pdf(image, label, confidence, info["treatment"], info["prevention"], fname)
            st.download_button("📥 Download Full Report (PDF)", pdf, "plant_report.pdf", use_container_width=True)
        else:
            combined_pdf = FPDF()
            combined_pdf.set_margins(20, 20, 20)
            for image, label, confidence, info, fname in all_results:
                build_pdf_page(combined_pdf, image, label, confidence, info["treatment"], info["prevention"], fname)
            pdf_bytes = combined_pdf.output(dest='S').encode('latin-1')
            st.download_button(
                f"📥 Download Combined Report ({len(all_results)} images)",
                pdf_bytes, "plant_report_combined.pdf", use_container_width=True
            )

# ----------- CAMERA -----------
elif option == "📷  Camera":
    cam = st.camera_input("Take a photo")
    if cam:
        image = Image.open(cam)
        st.markdown('<div class="section-label">Analysis Result</div>', unsafe_allow_html=True)
        label, confidence, info = predict_image(image)
        render_result(image, label, confidence, info)

# ----------- LIVE DETECTION -----------
elif option == "🎥  Live Detection":
    st.markdown('<div class="section-label">Live Camera Feed</div>', unsafe_allow_html=True)
    webrtc_ctx = webrtc_streamer(
        key="plant-detection",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
    if webrtc_ctx.video_processor:
        if st.button("📸 Capture & Analyze Frame", use_container_width=True):
            frame = webrtc_ctx.video_processor.latest_frame
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                st.markdown('<div class="section-label">Captured Frame Analysis</div>', unsafe_allow_html=True)
                label, confidence, info = predict_image(image)
                render_result(image, label, confidence, info)
            else:
                st.warning("No frame captured yet. Please wait for the camera to load.")

# ----------- FOOTER -----------
st.markdown("""
<div class="footer">
    🌿 &nbsp; Plant AI &nbsp;·&nbsp; Powered by Deep Learning &nbsp;·&nbsp; Made by 49,50,51
</div>
""", unsafe_allow_html=True)