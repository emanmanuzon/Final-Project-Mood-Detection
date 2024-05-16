import streamlit as st
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import av 

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

@st.cache_data(experimental_allow_widgets=True)
class VideoTransformer:
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        faces = face_cascade.detectMultiscale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.1,3)

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(img, format='bgr24')



def load_model():
  model=tf.keras.models.load_model('CNN_Model_7.h5')
  return model
model=load_model()
st.write("""
# Mood Detection"""
)
