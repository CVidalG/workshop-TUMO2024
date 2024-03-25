# Lesson 2 (week 2)

Objectives:

* 

## Music Generation

Theoritical presentation.

*Requirements*: YouTube musics, without noise (e.g. clapping, etc.)

### Set up

```bash
conda create --name prism python=3.8
pip install tensorflow tensorflow-metal
conda install -c conda-forge librosa
pip install natsort
pip install pydub
pip install keras-tuner
```

```bash
git clone https://github.com/rncm-prism/prism-samplernn
```

### Chunk creation

```bash
python chunk_audio.py \
  --input_file path/to/input.wav \
  --output_dir ./chunks \
  --chunk_length 8000 \
  --overlap 1000
```

### Training

```bash
python train.py \
  --id test \
  --data_dir ./chunks \
  --num_epochs 100 \
  --batch_size 128 \
  --checkpoint_every 5 \
  --output_file_dur 3 \
  --sample_rate 16000
```

### Music Generation

```bash
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
