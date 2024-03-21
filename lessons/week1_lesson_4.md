# Lesson 4 (week 1)

Objectives:

* Sharing data to increase training dataset
* Understanding training parameters and playing with them
* Coding a data split function
* Training a classification model

[ACCESS TO WORKSHOP GOOGLE DRIVE](https://drive.google.com/drive/folders/1ND3OpU4lX-P_aHAl6CWa-ZQZTlumOdGD?usp=sharing)

## Preliminary results of lesson 3 with a small dataset

**Training**:
```python
model = YOLO('yolov8n')
results = model.train(data='path/to/config.yaml') # default configuration
```

Training parameters we can use:
* `imgsz`: to change input image size
* `dropout`: to avoid overfitting
* `epochs`: to increase/decrease number of epochs
* `lr0`: to change initial learning rate

For a full list of parameters:
[Click to see YOLOv8 documentation](https://docs.ultralytics.com/modes/train/#train-settings)

💡 Maybe have a look to data augment parameters? Something relevant for the task?

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

**Postprocessing Python task**: Need to sort detections from top-to-bottom

Some tips:
1. predict `lines` on a test image;
2. load in python the txt file with detected bouding-boxes;
3. parse the txt file to extract bouding-boxes: each line correspond to a detection `class_id x y width height`
4. convert `x` `y` `width` and `height` into a shapely compatible format
5. extract centroïd from shapely object
6. sort detections according to centroïds, from top to bottom
7. crop and export image with relevant filename (e.g. : 1_..., 2_...) using PILLOW


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
