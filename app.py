import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Skin Cancer Classification", page_icon="🔬")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("skin_cancer_model.keras")

model = load_model()

st.title("Skin Cancer Classification")
st.write("CNN Based Image Classification System")

st.warning("This is an educational project, not a medical diagnostic tool.")

uploaded_file = st.file_uploader(
    "Upload Skin Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Skin Image", use_container_width=True)

    if st.button("Predict"):
        image = image.resize((128, 128))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        prediction = model.predict(image_array)
        score = float(prediction[0][0])

        if score >= 0.5:
            st.error("Model Prediction: Malignant")
        else:
            st.success("Model Prediction: Benign")

        st.write("Model Score:", round(score, 4))

        st.warning(
            "This result is experimental and must not be used "
            "for medical decisions. Consult a qualified healthcare professional."
        )

st.subheader("About Project")
st.write("""
This project uses a Convolutional Neural Network (CNN)
to classify skin images into Benign and Malignant categories.
""")
