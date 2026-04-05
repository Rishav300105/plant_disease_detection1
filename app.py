import streamlit as st
st.set_page_config(page_title="Plant AI", layout="centered")

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from fpdf import FPDF
import json
import tempfile
import os
import gdown   # ✅ added

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

# ----------- HIDE SIDEBAR -----------
st.markdown("""
<style>
[data-testid="stSidebar"] {display: none;}
</style>
""", unsafe_allow_html=True)

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

# ----------- LEAF CHECK FUNCTION -----------
def is_leaf(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    green_pixels = np.sum(mask > 0)
    total_pixels = frame.shape[0] * frame.shape[1]

    ratio = green_pixels / total_pixels

    return ratio > 0.08   # more flexible

# ----------- PREDICT FUNCTION -----------
def predict_image(image):

    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    if not is_leaf(frame):
        return "Not a plant leaf ❌", 0, {
            "treatment": "Upload leaf image only",
            "prevention": "Avoid background noise"
        }

    img = image.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = np.reshape(img, [1, IMG_SIZE, IMG_SIZE, 3])

    prediction = model.predict(img, verbose=0)
    index = np.argmax(prediction)
    confidence = prediction[0][index] * 100

    label = class_names[index]

    info = disease_info.get(label, {
        "treatment": "No data available",
        "prevention": "No data available"
    })

    return label, confidence, info

# ----------- VIDEO PROCESSOR -----------
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.latest_frame = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.latest_frame = img

        # Leaf check
        if not is_leaf(img):
            cv2.putText(img, "Not Leaf ❌", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0,0,255), 2)
            return img

        # Prediction
        resized = cv2.resize(img, (224, 224))
        normalized = resized / 255.0
        input_img = normalized.reshape(1, 224, 224, 3)

        pred = model.predict(input_img, verbose=0)
        index = np.argmax(pred)
        confidence = pred[0][index] * 100

        label = class_names[index]

        text = f"{label} ({confidence:.1f}%)"

        cv2.putText(img, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0,255,0), 2)

        return img

# ----------- PDF FUNCTION -----------
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

    pdf.cell(200, 10, f"Disease: {clean_label}", ln=True)
    pdf.cell(200, 10, f"Confidence: {confidence:.2f}%", ln=True)

    pdf.ln(5)
    pdf.multi_cell(0, 10, f"Treatment: {treatment}")
    pdf.multi_cell(0, 10, f"Prevention: {prevention}")

    return pdf.output(dest='S').encode('latin-1')

# ----------- UI -----------
st.title("🌱 Plant Disease Detection AI")

option = st.radio("Choose Input:", ["Upload Image", "Camera", "Live Detection"])

# ----------- UPLOAD -----------
if option == "Upload Image":
    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if file:
        image = Image.open(file)
        st.image(image, width=300)

        label, confidence, info = predict_image(image)

        st.success(f"🌿 Disease: {label}")
        st.info(f"📊 Confidence: {confidence:.2f}%")

        st.subheader("💊 Treatment")
        st.write(info["treatment"])

        st.subheader("🛡️ Prevention")
        st.write(info["prevention"])

        pdf = generate_pdf(image, label, confidence,
                           info["treatment"], info["prevention"])

        st.download_button("📥 Download Report", pdf, "report.pdf")

# ----------- CAMERA -----------
elif option == "Camera":
    cam = st.camera_input("Capture")

    if cam:
        image = Image.open(cam)
        st.image(image, width=300)

        label, confidence, info = predict_image(image)

        st.success(f"🌿 Disease: {label}")
        st.info(f"📊 Confidence: {confidence:.2f}%")

        st.subheader("💊 Treatment")
        st.write(info["treatment"])

        st.subheader("🛡️ Prevention")
        st.write(info["prevention"])

# ----------- LIVE DETECTION -----------
elif option == "Live Detection":
    st.subheader("🎥 Real-Time Detection")

    webrtc_ctx = webrtc_streamer(
        key="plant-detection",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    st.markdown("### 📸 Capture & Analyze")

    if webrtc_ctx.video_processor:
        if st.button("Capture Frame", key="capture_btn"):

            frame = webrtc_ctx.video_processor.latest_frame

            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)

                st.image(image, width=300)

                label, confidence, info = predict_image(image)

                st.success(f"🌿 Disease: {label}")
                st.info(f"📊 Confidence: {confidence:.2f}%")

                st.subheader("💊 Treatment")
                st.write(info["treatment"])

                st.subheader("🛡️ Prevention")
                st.write(info["prevention"])