import streamlit as st
import tensorflow as tf

@st.cache_data(experimental_allow_widgets=True)
def load_model():
  model=tf.keras.models.load_model('moodmodel.h5')
  return model
model=load_model()

st.write("""
# Mood Detection"""
)