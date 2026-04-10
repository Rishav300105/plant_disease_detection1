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

# ----------- CUSTOM CSS -----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');

/* ---- Base ---- */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d1a0f;
    color: #e8f0e9;
}

.stApp {
    background: radial-gradient(ellipse at top left, #122a15 0%, #0d1a0f 50%, #080f09 100%);
    min-height: 100vh;
}

/* ---- Hide sidebar ---- */
[data-testid="stSidebar"] { display: none; }

/* ---- Hero Header ---- */
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
.hero-title span {
    color: #5fd068;
}
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

/* ---- Divider ---- */
.leaf-divider {
    text-align: center;
    color: #2d5c32;
    font-size: 1.2rem;
    letter-spacing: 8px;
    margin: 1.5rem 0;
}

/* ---- Radio / Tabs ---- */
div[role="radiogroup"] {
    display: flex;
    gap: 12px;
    justify-content: center;
    flex-wrap: wrap;
    margin: 1rem 0 2rem;
}
div[role="radiogroup"] label {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(95,208,104,0.2) !important;
    border-radius: 50px !important;
    padding: 10px 24px !important;
    cursor: pointer !important;
    font-size: 0.9rem !important;
    color: #a0c8a4 !important;
    transition: all 0.2s ease !important;
    white-space: nowrap !important;
}
div[role="radiogroup"] label:hover {
    background: rgba(95,208,104,0.1) !important;
    border-color: rgba(95,208,104,0.5) !important;
    color: #5fd068 !important;
}
div[role="radiogroup"] [data-checked="true"] label,
div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) {
    background: rgba(95,208,104,0.15) !important;
    border-color: #5fd068 !important;
    color: #5fd068 !important;
}

/* ---- Upload Zone ---- */
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
[data-testid="stFileUploader"] * {
    color: #7aad7f !important;
}

/* ---- Image display ---- */
[data-testid="stImage"] img {
    border-radius: 16px !important;
    border: 1px solid rgba(95,208,104,0.15) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4) !important;
}

/* ---- Result Cards ---- */
.result-card {
    background: linear-gradient(135deg, rgba(18,42,21,0.9) 0%, rgba(13,26,15,0.9) 100%);
    border: 1px solid rgba(95,208,104,0.2);
    border-radius: 20px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: -50%; right: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at top right, rgba(95,208,104,0.05) 0%, transparent 60%);
    pointer-events: none;
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
    transition: width 0.8s ease;
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
.info-card-icon {
    font-size: 1.4rem;
    margin-bottom: 0.5rem;
}
.info-card-title {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #5fd068;
    font-weight: 500;
    margin-bottom: 0.4rem;
}
.info-card-text {
    font-size: 0.88rem;
    color: #c5d8c7;
    line-height: 1.5;
}

/* ---- Healthy / Disease Badge ---- */
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

/* ---- Streamlit Alerts ---- */
.stAlert {
    border-radius: 14px !important;
    border: none !important;
    background: rgba(95,208,104,0.08) !important;
}

/* ---- Buttons ---- */
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

/* ---- Section headings ---- */
h3, .stSubheader {
    font-family: 'Playfair Display', serif !important;
    color: #c8e6cb !important;
}

/* ---- Camera input ---- */
[data-testid="stCameraInput"] {
    border-radius: 16px !important;
    overflow: hidden !important;
}

/* ---- Footer ---- */
.footer {
    text-align: center;
    padding: 3rem 1rem 2rem;
    color: #2d5c32;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
}

/* ---- Section label ---- */
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
</style>
""", unsafe_allow_html=True)

# ----------- MODEL -----------
import gdown
import os

# ----------- MODEL LOAD (WITH GOOGLE DRIVE - FIXED) -----------
MODEL_PATH = "plant_disease_model.h5"

FILE_ID = "1MKW4o4Ux-8uMcO7QukDPwwa2GgLW01dq"
url = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):
    st.warning("⬇️ Downloading model from Google Drive...")

    try:
        gdown.download(url, MODEL_PATH, quiet=False)

        # ✅ check if file is valid (not html)
        if os.path.getsize(MODEL_PATH) < 1000000:
            st.error("❌ Downloaded file is corrupted!")
            os.remove(MODEL_PATH)
            st.stop()

        st.success("✅ Model downloaded successfully!")

    except Exception as e:
        st.error("❌ Auto-download failed!")
        st.write(e)
        st.stop()

# ----------- LOAD MODEL SAFELY -----------
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error("❌ Model loading failed (file corrupted)")
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
        return "Not a plant leaf ❌", 0, {
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
            cv2.putText(img, "Not a leaf", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            return img
        resized = cv2.resize(img, (224, 224))
        normalized = resized / 255.0
        input_img = normalized.reshape(1, 224, 224, 3)
        pred = model.predict(input_img, verbose=0)
        index = np.argmax(pred)
        confidence = pred[0][index] * 100
        label = class_names[index]
        cv2.putText(img, f"{label} ({confidence:.1f}%)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80,220,90), 2)
        return img

# ----------- PDF -----------
def generate_pdf(image, label, confidence, treatment, prevention):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, "Plant Disease Report", ln=True, align="C")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        img_path = tmp.name
    pdf.image(img_path, x=55, y=30, w=100)
    pdf.set_y(140)
    clean_label = label.encode("latin-1", "ignore").decode("latin-1")
    if "healthy" in clean_label.lower():
        pdf.cell(200, 10, "Status: Healthy Plant", ln=True)
    else:
        pdf.cell(200, 10, f"Disease: {clean_label}", ln=True)
    pdf.cell(200, 10, f"Confidence: {confidence:.2f}%", ln=True)
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"Treatment: {treatment}")
    pdf.multi_cell(0, 10, f"Prevention: {prevention}")
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

# ----------- HEADER -----------
st.markdown("""
<div class="hero-header">
    <div class="hero-subtitle">AI-Powered Diagnostics</div>
    <div class="hero-title">Plant <span>Disease</span><br>Detection</div>
    <div class="leaf-divider">✦ ✦ ✦</div>
    <div class="hero-desc">Upload a photo of your plant leaf for instant AI diagnosis, treatment recommendations, and prevention tips.</div>
</div>
""", unsafe_allow_html=True)

# ----------- MODE SELECTOR -----------
st.markdown('<div class="section-label">Choose Input Mode</div>', unsafe_allow_html=True)
option = st.radio("", ["📤  Upload Image", "📷  Camera", "🎥  Live Detection"], horizontal=True, label_visibility="collapsed")

st.markdown("<br>", unsafe_allow_html=True)

# ----------- UPLOAD -----------
if option == "📤  Upload Image":
    file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if file:
        image = Image.open(file)
        st.markdown('<div class="section-label">Analysis Result</div>', unsafe_allow_html=True)
        label, confidence, info = predict_image(image)
        render_result(image, label, confidence, info)
        st.markdown("<br>", unsafe_allow_html=True)
        pdf = generate_pdf(image, label, confidence, info["treatment"], info["prevention"])
        st.download_button("📥 Download Full Report (PDF)", pdf, "plant_report.pdf", use_container_width=True)

# ----------- CAMERA -----------
elif option == "📷  Camera":
    cam = st.camera_input("")
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
    🌿 &nbsp; Plant AI &nbsp;·&nbsp; Powered by Deep Learning &nbsp;·&nbsp; Built with Streamlit
</div>
""", unsafe_allow_html=True)