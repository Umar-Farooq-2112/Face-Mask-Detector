import streamlit as st
import cv2
import numpy as np
import pickle

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

with open("FaceDetector.pkl",'rb') as file:
    model = pickle.load(file)


if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image =cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    input = cv2.resize(image,(128,128))
    input = input/255
    input = np.reshape(input,[1,128,128,3])

    res = model.predict(input)
    print(res)
    res = np.argmax(res)
    
    if res == 1:
        st.write("Mask Detected")
    else:
        st.write("No Mask Detected")