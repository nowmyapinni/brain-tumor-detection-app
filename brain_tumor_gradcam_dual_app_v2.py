
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import gdown
import os

# Download models from Google Drive
fusion_model_path = "brain_tumor_fusion_model.h5"
cnn_model_path = "cnn_only_model.h5"

if not os.path.exists(fusion_model_path):
    gdown.download("https://drive.google.com/uc?id=1861aCqx_bvXRbz7QgR-v4tjSzlFkiTQi", fusion_model_path, quiet=False)

if not os.path.exists(cnn_model_path):
    gdown.download("https://drive.google.com/uc?id=1ZxDdaTUVpKMsvYw7mvABdekGI_8SSpQC", cnn_model_path, quiet=False)

# Load both models
fusion_model = load_model(fusion_model_path)
cnn_model = load_model(cnn_model_path)

st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠")
st.markdown("<h1 style='text-align: center;'>🧠 Brain Tumor Detection with GradCAM</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a Brain MRI Image", type=["jpg", "jpeg", "png"])

age = st.slider("Patient Age", 10, 90, 25)
gender = st.radio("Gender", ["Male", "Female"])
headache = st.radio("Headache", ["Yes", "No"])

def preprocess_tabular(age, gender, headache):
    gender_encoded = 1 if gender == "Male" else 0
    headache_encoded = 1 if headache == "Yes" else 0
    return np.array([[age, gender_encoded, headache_encoded]])

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(original_img, heatmap):
    img = np.uint8(255 * original_img)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
    return superimposed_img

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded MRI", use_column_width=True)
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    tab_input = preprocess_tabular(age, gender, headache)

    fusion_pred = fusion_model.predict([img_input, tab_input])[0][0]
    fusion_label = "Tumor" if fusion_pred > 0.5 else "No Tumor"
    confidence = fusion_pred if fusion_pred > 0.5 else 1 - fusion_pred

    st.markdown(f"### 🔮 Fusion Model Prediction: **{fusion_label}**")
    st.markdown(f"### 🔍 Confidence Score: **{confidence:.2f}**")

    if st.button("Show GradCAM from CNN-only Model (Better Focus)"):
        cnn_pred = cnn_model.predict(img_input)[0][0]
        st.markdown(f"### 🧠 CNN-only Model: Prediction = {'Tumor' if cnn_pred > 0.5 else 'No Tumor'}")
        heatmap = make_gradcam_heatmap(img_input, cnn_model, last_conv_layer_name="conv2d_3")
        heatmap_img = overlay_heatmap(img_array, heatmap)
        st.image(heatmap_img, caption="🔥 GradCAM from CNN-only Model", use_column_width=True)
