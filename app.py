
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page settings
st.set_page_config(
    page_title="Public Place Cleanliness AI",
    page_icon="🧹"
)

st.title("Public Place Cleanliness Detection")
st.write("Upload a public place image to check its cleanliness.")

# Load trained CNN model
model = tf.keras.models.load_model(
    "models/cleanliness_model.keras"
)

# Class names
class_names = ["Clean", "Dirty"]

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict Cleanliness"):

        # Image preprocessing
        image = image.resize((128, 128))

        image_array = np.array(image, dtype="float32")
        image_array = np.expand_dims(image_array, axis=0)

        # Model prediction
        prediction = model.predict(image_array, verbose=0)

        result = np.argmax(prediction[0])
        confidence = prediction[0][result] * 100

        # Show result
        st.subheader("Prediction Result")

        st.success(class_names[result])

        st.write(
            f"Confidence: {confidence:.2f}%"
        )