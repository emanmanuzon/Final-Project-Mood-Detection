import streamlit as st
import tensorflow as tf

@st.cache_data(experimental_allow_widgets=True)
st.write("""
# Mood Detection"""
)
file=st.file_uploader("Choose weather photo from computer",type=["jpg","png"])
