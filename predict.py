import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load model
model = load_model("plant_disease_model.h5")

IMG_SIZE = 224

# Class names (IMPORTANT: match your dataset folders)
import json

# load class indices
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# reverse mapping
class_names = list(class_indices.keys())
class_names = sorted(class_names, key=lambda x: class_indices[x])

# Disease info (treatment + prevention)
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

# Load image
img_path = "test.jpg"   # change image name if needed
img = cv2.imread(img_path)

img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0
img = np.reshape(img, [1, IMG_SIZE, IMG_SIZE, 3])

# Prediction
prediction = model.predict(img, verbose=0)

# Get result
index = np.argmax(prediction)
predicted_class = class_names[index]
confidence = prediction[0][index] * 100

# Get treatment & prevention
info = disease_info.get(predicted_class, {
    "treatment": "No data available",
    "prevention": "No data available"
})

# Output
print("\n🌿 Disease:", predicted_class)
print(f"📊 Confidence: {confidence:.2f}%")

print("\n💊 Treatment:")
print(info["treatment"])

print("\n🛡️ Prevention:")
print(info["prevention"])