import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.set_page_config(page_title="Real-time Face Detection", page_icon="🎥", layout="wide")

st.title("Face Detection App")
st.markdown("Detect faces from **uploaded images**, **camera capture**, or **real-time video** feed.")

st.sidebar.title("**Select Mode**")
option = st.sidebar.radio("Choose input type:", ("Upload Image", "Camera Capture", "Real-time Video"))


if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_np = np.array(image)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img_np, (x, y), (x+w, y+h), (0, 255, 0), 2)
        col1,col2 = st.columns(2)
        with col1:
            st.write("### Uploaded Image:")
            st.image(image, width="stretch")
        with col2:
            st.write("### After Face Detection:")
            st.image(img_np, caption=f"Detected {len(faces)} face(s)", width="stretch")

elif option == "Camera Capture":
    img_file_buffer = st.camera_input("Take a photo")

    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        img_np = np.array(image)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img_np, (x, y), (x+w, y+h), (0, 255, 0), 2)

        st.image(img_np, caption=f"Detected {len(faces)} face(s)", width="stretch")

elif option == "Real-time Video":
    st.markdown("**Live Face Detection** from your webcam")

    class VideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.putText(img,"Face Detected",(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 0, 255),1)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)

            return frame.from_ndarray(img, format="bgr24")

    webrtc_streamer(key="face-detect", video_processor_factory=VideoProcessor)

st.markdown("---")
