import streamlit as st
from streamlit_webrtc import VideoTransformer, webrtc_streamer

# Define a custom video transformer to display the video feed
class VideoTransformerBase(VideoTransformer):
    def transform(self, frame):
        return frame

# Title of the app
st.title("Streamlit Camera Feed")

# Define the WebRTC streamer
webrtc_ctx = webrtc_streamer(
    key="example",
    video_transformer_factory=VideoTransformerBase,
    async_transform=True,
)

# Display the video feed
if webrtc_ctx.video_transformer:
    webrtc_ctx.video_transformer.video_sink.set_layout("contain")
    st.video(webrtc_ctx)
else:
    st.warning("No video stream found.")

# Optional: Add any additional Streamlit components or UI elements
