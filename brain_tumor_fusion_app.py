import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import gdown

# ✨ Page setup
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠")
st.markdown("<h1 style='text-align: center;'>🧠 Brain Tumor Detection with GradCAM</h1>", unsafe_allow_html=True)

# ✨ Upload image
uploaded_file = st.file_uploader("Upload a Brain MRI Image", type=["jpg", "jpeg", "png"])

# ✨ Input metadata
age = st.slider("Patient Age", 10, 90, 25)
gender = st.radio("Gender", ["Male", "Female"])
headache = st.radio("Headache", ["Yes", "No"])

def preprocess_tabular(age, gender, headache):
    gender_encoded = 1 if gender == "Male" else 0
    headache_encoded = 1 if headache == "Yes" else 0
    return np.array([[age, gender_encoded, headache_encoded]])

# ✨ Download and load fusion model
fusion_model_path = "brain_tumor_app.h5"
if not os.path.exists(fusion_model_path):
    fusion_url = "https://drive.google.com/uc?id=1861aCqx_bvXRbz7QgR-v4tjSzlFkiTQi"
    gdown.download(fusion_url, fusion_model_path, quiet=False)
model = load_model(fusion_model_path)

# 🌟 GradCAM for fusion model
def make_gradcam_heatmap(img_array, tabular_array, model, last_conv_layer_name="conv2d_7"):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([img_array, tabular_array])
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# 🌟 Overlay helper

def overlay_heatmap(original_img, heatmap):
    img = np.uint8(255 * original_img)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
    return superimposed_img

# 💫 Main logic
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB').resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_input = np.expand_dims(img_array, axis=0)
    tab_input = preprocess_tabular(age, gender, headache)

    prediction = model.predict([img_input, tab_input])[0][0]
    label = "Tumor" if prediction > 0.5 else "No Tumor"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.image(img, caption="🗃 Uploaded MRI", use_column_width=True)
    st.markdown(f"### 🔮 Prediction: **{label}**")
    st.markdown(f"### 🔍 Confidence Score: **{confidence:.2f}**")

    heatmap = make_gradcam_heatmap(img_input, tab_input, model)
    overlay = overlay_heatmap(img_array, heatmap)
    st.image(overlay, caption="🔥 Fusion Model GradCAM", use_column_width=True)

    # ✨ Load CNN-only model for extra GradCAM
    cnn_model_path = "cnn_only_model.h5"
    if not os.path.exists(cnn_model_path):
        cnn_url = "https://drive.google.com/uc?id=1ZxDdaTUVpKMsvYw7mvABdekGI_8SSpQC"
        gdown.download(cnn_url, cnn_model_path, quiet=False)
    cnn_model = load_model(cnn_model_path)

    if st.button("🔥 Try GradCAM from CNN-only Model (More Focused)"):
        def make_cnn_gradcam_heatmap(img_array, model, last_conv_layer_name="conv2d_2"):
            grad_model = tf.keras.models.Model(
                [model.input],
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

        cnn_heatmap = make_cnn_gradcam_heatmap(img_input, cnn_model)
        cnn_overlay = overlay_heatmap(img_array, cnn_heatmap)
        st.image(cnn_overlay, caption="🌟 CNN-only GradCAM (Focused Inside Tumor)", use_column_width=True)
