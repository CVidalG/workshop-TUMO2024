# Lesson 1 (week 2)

Objectives:

* Music generation using OMR and YOLO models;
* Music sheet classification;
* Building a dataset for music generation;
* Webapp development using Python.

## Music Generation using OMR

We have setup your models inside the OMR environment.

```bash
conda activate omr
```

### Checking setup

Download [piano_super_easy.jpg](assets/piano_super-easy.jpg) image.

#### Detecting and predicting notes:

```bash
oemer path/to/piano/super/easy.jpg
```

It involves three models: (i) symbol detection (see week1), (ii) staff lines and rythm dectection, (iii) combining results.

You should now have:
* piano_super-easy.musicxml = the recognition result stored in a musicxml file.
* piano_super-easy_teaser.png = a preview of the detection

#### Converting to MIDI file

```bash
python -c "import muspy ; music = muspy.read_musicxml('/path/to/musicXML/file.musicxml') ; muspy.write_midi('/path/to/export/midi/file.midi', music, backend='pretty_midi')"
```

#### Using GarageBand to remix it a bit

Open the MIDI file with GarageBand, play the music and try everything.

### Now, the same with a YOLO crop ?

The answer to generate crops of lines with your YOLO model is somewhere between lessons 4 and 5 of the week 1.

```bash
model = YOLO('path_to_your_model')
results = model.predict(...)
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


## WebApp development using YOLO and streamlit

In omr environment :

```bash
pip install streamlit
```

### Basic image detection app

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

    model_name = st.sidebar.selectbox("Select Model", ["yolov8n"])
    model = load_model(model_name)
    confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file).convert("RGB")

        if st.sidebar.button('Detect Objects'):
            res = model.predict(uploaded_image, conf=confidence, imgsz=640)

            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            with col2:
                st.image(res[0].plot(), caption='Detected Image', use_column_width=True)
        else:
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

if __name__ == "__main__":
    main()
```