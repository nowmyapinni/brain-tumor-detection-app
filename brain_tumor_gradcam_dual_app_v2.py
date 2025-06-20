
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

