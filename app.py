import streamlit as st
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# =====================================================================================
# CONFIG
# =====================================================================================
st.set_page_config(page_title="TB Classification", layout="wide")
CLASS_NAMES = ["TB", "Non-TB", "Normal"]


# =====================================================================================
# LOAD MODELS
# =====================================================================================
@st.cache_resource
def load_models():
    base_path = os.path.dirname(__file__)
    model_div_path = os.path.join(base_path, "model", "ensemble_DIV_canny.keras")
    model_vd_path  = os.path.join(base_path, "model", "ensemble_VD_CLAHE.keras")

    model_div = load_model(model_div_path, compile=False)
    model_vd  = load_model(model_vd_path, compile=False)

    return model_div, model_vd

MODEL_DIV, MODEL_VD = load_models()


# =====================================================================================
# PREPROCESSING
# =====================================================================================
def apply_clahe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def apply_canny_from_clahe(clahe_img):
    blurred = cv2.GaussianBlur(clahe_img, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(thresh, 100, 200)
    return edges


def prepare(image, size=416):
    img = cv2.resize(image, (size, size))

    if len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)

    img = img.astype("float32") / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img


# =====================================================================================
# PREDICTION
# =====================================================================================
def predict(model, image):
    pred = model.predict(image)[0]
    idx = np.argmax(pred)
    return CLASS_NAMES[idx], pred


# =====================================================================================
# STREAMLIT UI
# =====================================================================================
st.title("🩺 TB Classification with Dual-Model Ensemble")

st.write("""
### Preprocessing:
1. CLAHE  
2. CLAHE + Canny  

### Prediction Path:
- CLAHE → Model VD  
- Canny → Model DIV  

### 📌 Spesifikasi Gambar yang Diperbolehkan
- Format **PNG**
- Orientasi **Potrait**
- Jenis gambar **CXR (Chest X-Ray)**
""")


# ----------------------------------------------------------
# FILE UPLOADER (with size limit)
# ----------------------------------------------------------
uploaded = st.file_uploader("Upload CXR Image (PNG Only, Max 5MB)", type=["png"])

if uploaded is not None:

    # Validasi ukuran file 5MB
    if uploaded.size > 5 * 1024 * 1024:
        st.error("❌ Ukuran file melebihi 5MB. Harap upload file yang lebih kecil.")
        st.stop()

    # Convert file ke array
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # ---------- PREPROCESS ----------
    clahe_img = apply_clahe(image)
    canny_img = apply_canny_from_clahe(clahe_img)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)
    with col2:
        st.image(clahe_img, caption="CLAHE Result", use_container_width=True)
    with col3:
        st.image(canny_img, caption="CLAHE + Canny Result", use_container_width=True)

    # ------------------------------------------------------
    # BUTTON PREDICT
    # ------------------------------------------------------
    if st.button("Predict"):

        clahe_ready = prepare(clahe_img)
        canny_ready = prepare(canny_img)

        label_clahe, prob_clahe = predict(MODEL_VD, clahe_ready)
        label_canny, prob_canny = predict(MODEL_DIV, canny_ready)

        st.subheader("Prediction Results")

        colA, colB = st.columns(2)
        with colA:
            st.markdown("### 🔵 CLAHE → Model VD")
            st.success(f"Prediksi: **{label_clahe}**")

        with colB:
            st.markdown("### 🔴 CLAHE + Canny → Model DIV")
            st.success(f"Prediksi: **{label_canny}**")

        # ------------------------------------------------------
        # FINAL ENSEMBLE DECISION
        # ------------------------------------------------------
        st.subheader("Final Decision (Ensemble Output)")

        if label_clahe == label_canny:
            final_label = label_clahe
            st.success(f"🟢 Kedua model setuju → **{final_label}**")

        else:
            conf_clahe = np.max(prob_clahe)
            conf_canny = np.max(prob_canny)

            if conf_clahe > conf_canny:
                final_label = label_clahe
                st.warning(f"⚠️ Model berbeda → Dipilih: CLAHE (VD) → **{final_label}**")
            else:
                final_label = label_canny
                st.warning(f"⚠️ Model berbeda → Dipilih: Canny (DIV) → **{final_label}**")

        st.success(f"### 🟩 Final Classification: **{final_label}**")
