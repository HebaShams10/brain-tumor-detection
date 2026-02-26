import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ======= Load the trained model =======
@st.cache_resource  
def load_model():
    model = tf.keras.models.load_model("brain_tumor_resnet50.keras")
    return model

model = load_model()

# Labels
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Streamlit page settings
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("🧠 Brain Tumor MRI Tumor Detection")

# Upload image
uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded MRI Image", use_column_width=True)

    # Preprocess for model
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    preds = model.predict(img_array)
    pred_class = class_labels[np.argmax(preds)]
    confidence = np.max(preds)

    # Show result
    st.subheader("Prediction:")
    st.markdown(f"**{pred_class}** (Confidence: `{confidence:.2%}`)")

    # Show probabilities for all classes
    st.subheader("Prediction Probabilities")
    prob_dict = {class_labels[i]: float(preds[0][i]) for i in range(4)}
    st.bar_chart(prob_dict)