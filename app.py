import streamlit as st
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2

@st.cache_data(experimental_allow_widgets=True)
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        return frame

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def load_model():
  model=tf.keras.models.load_model('CNN_Model_7.h5')
  return model
model=load_model()
st.write("""
# Mood Detection"""
)
