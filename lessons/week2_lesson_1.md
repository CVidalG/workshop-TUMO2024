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

