import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image, ImageOps 


def load_model():
    print("Loading model...")
    model = tf.keras.models.load_model('moodmodel.h5')
    print("Model loaded successfully!")
    return model

# Function to detect faces in an image
def detect_faces(image):
    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around the detected faces and extract ROIs
    rois = []
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi = image[y:y+h, x:x+w]
        rois.append(roi)
    
    return image, len(faces), rois

def import_and_predict(image_data,model):
    size=(48,48)
    image=ImageOps.fit(image_data,size)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction

# Streamlit app
st.title("Mood Detection")

file = st.file_uploader("Choose a photo from your computer", type=["jpg", "png"])

if file is not None:
    # Read the uploaded image
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), -1)  # Use np.frombuffer to handle binary data

    # Display the original image
    st.image(image, channels="BGR", caption='Original Image')

    # Detect faces in the image
    image_with_faces, num_faces, rois = detect_faces(image)

    # Display the image with detected faces
    st.image(image_with_faces, channels="BGR", caption=f'Image with {num_faces} face(s) detected')

    # Display the ROIs of the detected faces
    for i, roi in enumerate(rois):
        st.image(roi, channels="BGR", caption=f'Region of Interest {i+1}')

    for i, roi in enumerate(rois):
        # Make prediction for the ROI
        prediction = import_and_predict(Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)), model)



