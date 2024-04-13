import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import os
import subprocess
import time

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
    model = load_model('models/' + model_name)

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

def music_reading():
    st.sidebar.header("Reading music from music sheets")

    demo_images_to_audio = {
        "Չինար ես (Կոմիտաս)": {"image": "demo/score3.png", "audio": "audio/score3.mp3"},
        "Յոթ պար": {"image": "demo/score_1.jpg", "audio": "audio/score_1.mp3"},
    }

    img_file = st.sidebar.file_uploader("Upload Sheet Music Image", type=["jpg", "jpeg", "png"])

    demo_image_choice = st.sidebar.selectbox('Choose a demo image', list(demo_images_to_audio.keys()))
    demo_image_path = demo_images_to_audio[demo_image_choice]["image"]
    demo_audio_path = demo_images_to_audio[demo_image_choice]["audio"]
    demo_image = Image.open(demo_image_path).convert("RGB")

    if img_file is not None:
        uploaded_image = Image.open(img_file).convert("RGB")
    else:
        uploaded_image = demo_image


    if st.sidebar.button('Read Music sheet'):

        if demo_audio_path is not None:
            with st.spinner('Please wait...'):
                time.sleep(5)
                with open(demo_audio_path, "rb") as audio_file:
                    st.audio(audio_file.read(), format='audio/wav')
        else:
            img_path = os.path.join(uploaded_image.name)
            with open(img_path, "wb") as f:
                f.write(uploaded_image.getbuffer())

            # Step 1: Convert image to MusicXML using oemer
            musicxml_path = img_path.replace('.jpg', '.musicxml').replace('.jpeg', '.musicxml').replace('.png', '.musicxml')
            subprocess.run(["oemer", img_path], check=True)

            # Step 2: Convert MusicXML to MIDI
            midi_path = musicxml_path.replace('.musicxml', '.midi')
            subprocess.run(["python", "-c", f"import muspy; music = muspy.read_musicxml('{musicxml_path}'); muspy.write_midi('{midi_path}', music, backend='pretty_midi')"], check=True)

            # Play the MIDI file
            if os.path.exists(midi_path):
                with open(midi_path, "rb") as midi_file:
                    st.audio(midi_file.read(), format='audio/ogg')
            else:
                st.write("Failed to generate MIDI file.")
                
            ## You will need to convert midi to wav or to mp3 to run it in Streamlit

    st.image(uploaded_image, caption='Input Image', use_column_width=True)

def main():

    st.title('TUMO workshop - Reading and computing music with AI')
    task = ['Select a task...', 'Music Detection', 'Reading Music', 'Music Generation']
    st.write('This workshop is about Music Generation and Music Detection. How can we use Artificial Intelligence to read and promote Armenian Music Heritage ?')
    
    st.sidebar.image('assets/OIG1vE3TF0tjVWcQOy3k_PWL.jpeg')
    st.sidebar.write('Code source and models: https://github.com/CVidalG/workshop-TUMO2024')

    task_menu = st.selectbox('Tasks', options=task, index=0)


    if task_menu == 'Music Detection':
        music_detection()
    elif task_menu == 'Reading Music':
        music_reading()
    elif task_menu == 'Music Generation':
        music_generation()
    elif task_menu == 'Select a task...':
        col1, col2 = st.columns(2)

        with col1:
            st.image('assets/tumo.png', width=200)
        with col2:
            st.image('assets/calfa.png', width=160)



if __name__ == "__main__":
    main()