import streamlit as st
import cv2
import numpy as np

# Function to detect faces in an image
def detect_faces(image):
    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return image, len(faces)

# Streamlit app
st.title("Mood Detection")

file = st.file_uploader("Choose a photo from your computer", type=["jpg", "png"])

if file is not None:
    # Read the uploaded image
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), -1)  # Use np.frombuffer to handle binary data

    # Display the original image
    st.image(image, channels="BGR", caption='Original Image')

    # Detect faces in the image
    image_with_faces, num_faces = detect_faces(image)

    # Display the image with detected faces
    st.image(image_with_faces, channels="BGR", caption=f'Image with {num_faces} face(s) detected')
