# Lesson 2 (week 2)

Objectives:

* Extend functionalities of our WebApp
* Creating a dataset of sounds
* Training a Music Generation model

## WebApp development using YOLO and streamlit

See Week 2, lesson 1, for the base code.

Extend the code by adding class filtering.

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