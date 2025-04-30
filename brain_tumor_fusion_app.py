import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.cm as cm
import gdown
gdown.download("https://drive.google.com/uc?id=1861aCqx_bvXRbz7QgR-v4tjSzlFkiTQi", "model.h5", quiet=False)
model = load_model("model.h5")

# Streamlit UI
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠")
st.markdown("<h1 style='text-align: center;'>🧠 Brain Tumor Detection with GradCAM</h1>", unsafe_allow_html=True)

# Upload image
uploaded_file = st.file_uploader("Upload a Brain MRI Image", type=["jpg", "jpeg", "png"])

# Input metadata
age = st.slider("Patient Age", 10, 90, 25)
gender = st.radio("Gender", ["Male", "Female"])
headache = st.radio("Headache", ["Yes", "No"])

# Preprocess tabular data
def preprocess_tabular(age, gender, headache):
    gender_encoded = 1 if gender == "Male" else 0
    headache_encoded = 1 if headache == "Yes" else 0
    return np.array([[age, gender_encoded, headache_encoded]])

# GradCAM function
def make_gradcam_heatmap(img_array, tabular_array, model, last_conv_layer_name="last_conv_layer"):
    grad_model = tf.keras.models.Model(
        [model.input],
        [model.get_layer(last_conv_layer_name).output, model.output]
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

# Overlay heatmap

def overlay_heatmap(original_img, heatmap):
    # Resize heatmap to match the image
    heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize((224, 224))
    heatmap_colored = cm.jet(np.array(heatmap_resized))[:, :, :3]
    superimposed = 0.6 * original_img + 0.4 * heatmap_colored
    return np.uint8(255 * superimposed)

# Prediction
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded MRI", use_column_width=True)
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    tab_input = preprocess_tabular(age, gender, headache)

    prediction = model.predict([img_input, tab_input])[0][0]
    label = "Tumor" if prediction > 0.5 else "No Tumor"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.markdown(f"### 🔮 Prediction: **{label}**")
    st.markdown(f"### 🔍 Confidence Score: **{confidence:.2f}**")

    # Generate and show GradCAM
    heatmap = make_gradcam_heatmap(img_input, tab_input, model)
    heatmap_img = overlay_heatmap(img_array, heatmap)
    st.image(heatmap_img, caption="🔥 GradCAM Tumor Heatmap", use_column_width=True)
