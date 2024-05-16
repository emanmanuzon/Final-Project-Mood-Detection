import streamlit as st
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import av 

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # You can pass the detected faces to your model for mood detection here
        return img

@st.cache_data(experimental_allow_widgets=True)
def load_model():
  model=tf.keras.models.load_model('CNN_Model_7.h5')
  return model
model=load_model()
st.write("""
# Weather Detection System"""
)

webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
