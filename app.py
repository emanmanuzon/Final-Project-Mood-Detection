from streamlit_webrtc import webrtc_streamer
import av
import streamlit as st
import cv2

st.title("Mood Detection")

file=st.file_uploader("Choose weather photo from computer",type=["jpg","png"])

