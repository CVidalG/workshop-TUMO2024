# Lesson 3 (week 2)

Objectives:

* Extend functionalities of our WebApp
* Understanding results of the Music Generation model
* Training a new Music Generation model

## WebApp development using YOLO and streamlit

Addition of menu and sound management

```python
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np


def load_model(model_name):
    model = YOLO(model_name)
    return model


def music_detection():
    st.sidebar.header("Music sheet detection")


    demo_images = {

        "Image 1": "demo/score_1.jpg",
        "Image 2": "demo/score_2.jpg",
        "Image 3": "demo/score3.png"
    }



    model_name = st.sidebar.selectbox("Select Model", ['MusicDetection_model.pt', "yolov8n"])
    model = load_model(model_name)

    classes_list = ['black', 'key', 'line', 'piano', 'tone', 'voice', 'white']
    selected_classes = st.sidebar.multiselect('select classes to predict', classes_list, default=classes_list)
    selected_classes_ids = [classes_list.index(cls) for cls in selected_classes]

    confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)


    demo_image_choice = st.sidebar.selectbox('Choose a demo image', list(demo_images.keys()))
    demo_image_path = demo_images[demo_image_choice]
    demo_image = Image.open(demo_image_path).convert("RGB")


    uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file).convert("RGB")
    else:
        uploaded_image = demo_image

    if st.sidebar.button('Detect Objects'):
        res = model.predict(uploaded_image, conf=confidence, imgsz=640, classes=selected_classes_ids)

        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        with col2:
            st.image(res[0].plot(labels=False), caption='Detected Image', use_column_width=True)
    else:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

def music_generation():
    st.sidebar.header("Music generation")

    # please upload a WAV file as input
    audio_file = st.file_uploader("Upload audio", type=["wav", "mp3"])

    if audio_file is not None:
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')


def main():

    st.title('TUMO workshop - Reading and computing music with AI')
    task = ['Select a task...', 'Music Detection', 'Music Generation']
    st.write('This workshop is about Music Generation and Music Detection. It takes in place in TUMO Yerevan.')
    
    task_menu = st.selectbox('Tasks', options=task, index=0)


    if task_menu == 'Music Detection':
        music_detection()
    elif task_menu == 'Music Generation':
        music_generation()
    elif task_menu == 'Select a task...':
        col1, col2 = st.columns(2)

        with col1:
            st.image('logo/Tumo-Logo.jpg', width=200)
        with col2:
            st.image('logo/logo-noir-texte-droite.png', width=200)




if __name__ == "__main__":
    main()
```