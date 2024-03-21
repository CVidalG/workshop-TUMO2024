# Lesson 4 (week 1)

Objectives:

* Sharing data to increase training dataset
* Understanding training parameters and playing with them
* Coding a data split function
* Training a classification model


## Preliminary results of lesson 3 with a small dataset

**Training**:
```python
model = YOLO('yolov8n')
results = model.train(data='path/to/config.yaml') # default configuration
```

Training parameters we can use:
* `imgsz`: to change input image size
* `dropout`: to use dropout to avoid overfitting

For a full list of parameters:
[Click to see YOLOv8 documentation](https://docs.ultralytics.com/modes/train/#train-settings)


**Prediction**:
```python
model = YOLO('path/to/my/best/model.pt')
results = model.predict(source='path/to/test/image')
```

Prediction parameters we can use:

* `classes`: list of int, to predict only specific classes
* `source=0` and `show=True`, to use webcam
* `save_txt`: to export object coordinates
* `save_crop`: to crop objects


## Datasets for Image Classification using YOLO

```python
model = YOLO('yolov8n-cls') # classification
results = model.train() # with params of your choice
```

**Dataset 1**: the CVC-MUSCIMA Database - writer identification task
[Download link](http://datasets.cvc.uab.es/muscima/CVCMUSCIMA_WI.zip)

**Dataset 2**: MUSCIMA++ - notes classification v0.9.1
[Download link](https://ufal.mff.cuni.cz/muscima/download)

**Preprocessing Python task 1**: Need to extract MUSCIMA++ images from XML annotations files.

**Preprocessing Python task 2**: Need to split datasets into train, val, test folders

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