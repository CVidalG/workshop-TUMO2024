# Lesson 4 (week 2)

Objectives:

* Extend functionalities of our WebApp by adding music sheet to audio;
* Understanding results of the Music Generation model;
* Music Classification using YOLO.

## WebApp

Starting from the code of week2 lesson3, add oemer pipeline.

## Music Classification using YOLO

**Dataset**: the CVC-MUSCIMA Database - writer identification task
[Download link](http://datasets.cvc.uab.es/muscima/CVCMUSCIMA_WI.zip)

### Data preprocessing

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

### Training

```python
model = YOLO('yolov8n-cls') # classification
results = model.train() # with params of your choice
```