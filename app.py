import streamlit as st
from keras.models import load_model
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('malaria_cnn-2.h5')

# Define function for image preprocessing
def preprocess_image(image):
    image = image.resize((64, 64))  # Resize image to 64x64
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define Streamlit UI
def main():
    st.title("Malaria Cell Image Classifier")
    st.write("Upload a malaria cell image for classification.")

    # Allow user to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(processed_image)

        # Display prediction result

        st.write("Prediction of Innfected ", prediction[0][0])

        st.write("Prediction of UnInnfected ",prediction[0][1])

if __name__ == '__main__':
    main()
