# Lesson 2 (week 2)

Objectives:

* Extend functionalities of our WebApp
* Creating a dataset of sounds
* Training a Music Generation model

## WebApp development using YOLO and streamlit

See Week 2, lesson 1, for the base code.

Extend the code by adding class filtering.

```python
import streamlit as st
from PIL import Image
from ultralytics import YOLO


def load_model(model_name):
    model = YOLO(model_name)
    return model


def main():
    st.header("My TUMO webapp")
    st.sidebar.header("Settings")


    demo_images = {
        "Image 1": "score_1.jpg",
        "Image 2": "score_2.jpg",
        "Image 3": "score_3.png"
    }

    model_name = st.sidebar.selectbox("Select Model", ["yolov8n", 'my_model.pt'])
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

if __name__ == "__main__":
    main()
```

## Music Generation

Theoritical presentation.

For this first experiment, we will use: [SampleRNN: An Unconditional End-to-End Neural Audio Generation Model](https://arxiv.org/abs/1612.07837)

Tensorflow version by [PRiSM](https://github.com/rncm-prism/prism-samplernn)

*Requirements*: YouTube musics, without noise (e.g. clapping, etc.)

### Set up on MacOS Ventura M1

```bash
conda create --name prism python=3.8
```
```bash
pip install tensorflow tensorflow-metal
```
```bash
conda install -c conda-forge librosa
```
```bash
pip install natsort
```
```bash
pip install pydub
```
```bash
pip install keras-tuner
```

and finally:

```bash
git clone https://github.com/rncm-prism/prism-samplernn
```

### Create your dataset

We need the following :

```bash
pip install yt-dlp
```

```bash
conda install conda-forge::ffmpeg
```

To download and convert to wav file :

```bash
yt-dlp [youtube_id] -x --audio-format wav --audio-quality 9 
```

Optional : to download and convert to wav file, with list.bat containing a list of YT urls

```bash
yt-dlp -x --audio-format wav --audio-quality 9 -i -a list.bat
```

### Chunk creation

```bash
python chunk_audio.py --input_file path/to/input.wav --output_dir ./chunks --chunk_length 8000 --overlap 1000
```

### Training

```bash
python train.py --id test --data_dir ./chunks --num_epochs 100 --batch_size 128 --checkpoint_every 1 --output_file_dur 3 --sample_rate 16000
```

### Music Generation

```bash
python generate.py --output_path path/to/out.wav --checkpoint_path ./logdir/path/to/model.ckpt-ID --config_file ./default.config.json --num_seqs 10 --dur 10 --sample_rate 16000 --seed path/to/seed.wav --seed_offset 500
```

## Music Classification using YOLO

**Dataset**: the CVC-MUSCIMA Database - writer identification task
[Download link](http://datasets.cvc.uab.es/muscima/CVCMUSCIMA_WI.zip)

Need to split the dataset into train, val, test folders.

From:
```
├── data/
│   ├── class_1/
│   ├── class_2/
│   ├── class_3/
│   ├── ...
│   ├── class_n/
```

to:
```
├── data/
│   ├── train/
|   |   |── class_1/ (70% of class_1 data)
|   |   |── class_2/ (70% of class_2 data)
|   |   |── ...
|   |   |── class_n/ (70% of class_n data)
│   ├── val
|   |   |── class_1/ (20% of class_1 data)
|   |   |── class_2/ (20% of class_2 data)
|   |   |── ...
|   |   |── class_n/ (20% of class_n data)
│   ├── test
|   |   |── class_1/ (10% of class_1 data)
|   |   |── class_2/ (10% of class_2 data)
|   |   |── ...
|   |   |── class_n/ (10% of class_n data)
```

In omr environment :

```bash
pip install scikit-learn
```

In a new notebook:

```python
import os
import shutil
from sklearn.model_selection import train_test_split

source_dir = 'path/to/your/data'
train_dir = 'path/to/your/train'
val_dir = 'path/to/your/val'
test_dir = 'path/to/your/test'

def split_data(source, train, val, test, initial_split_size=0.8, test_size=0.05):

    # Code here the split function

```
