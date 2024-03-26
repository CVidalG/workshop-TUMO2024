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

<<<<<<< HEAD
### Set up on MacOS Ventura M1
=======
### Set up
>>>>>>> c287e2d363870070198ab481f1950c4c31005dea

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
<<<<<<< HEAD
python chunk_audio.py --input_file path/to/input.wav --output_dir ./chunks --chunk_length 8000 --overlap 1000
=======
python chunk_audio.py \
  --input_file path/to/input.wav \
  --output_dir ./chunks \
  --chunk_length 8000 \
  --overlap 1000
>>>>>>> c287e2d363870070198ab481f1950c4c31005dea
```

### Training

```bash
<<<<<<< HEAD
python train.py --id test --data_dir ./chunks --num_epochs 100 --batch_size 128 --checkpoint_every 1 --output_file_dur 3 --sample_rate 16000
=======
python train.py \
  --id test \
  --data_dir ./chunks \
  --num_epochs 100 \
  --batch_size 128 \
  --checkpoint_every 5 \
  --output_file_dur 3 \
  --sample_rate 16000
>>>>>>> c287e2d363870070198ab481f1950c4c31005dea
```

### Music Generation

```bash
<<<<<<< HEAD
python generate.py --output_path path/to/out.wav --checkpoint_path ./logdir/path/to/model.ckpt-ID --config_file ./default.config.json --num_seqs 10 --dur 10 --sample_rate 16000 --seed path/to/seed.wav --seed_offset 500
```
=======
python generate.py \
  --output_path path/to/out.wav \
  --checkpoint_path ./logdir/path/to/model.ckpt-100 \
  --config_file ./default.config.json \
  --num_seqs 10 \
  --dur 10 \
  --sample_rate 16000 \
  --seed path/to/seed.wav \
  --seed_offset 500
```
>>>>>>> c287e2d363870070198ab481f1950c4c31005dea
