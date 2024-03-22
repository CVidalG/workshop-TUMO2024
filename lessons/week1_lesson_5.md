# Lesson 4 (week 1)

Objectives:

* Evaluating models
* Applying OMR on line crops / full music sheet (best strategy to choose)
* Using a music software to remix the generated audio
* Training a classification model

[ACCESS TO WORKSHOP GOOGLE DRIVE](https://drive.google.com/drive/folders/1ND3OpU4lX-P_aHAl6CWa-ZQZTlumOdGD?usp=sharing)

## Results of lesson 4 with the group dataset

All results and relevant informations are in the `runs/detect/train_ID` folder.

The `runs` folder is localized at the same level than your notebook.

### Dataset composition

<p align="center">
<img src="assets/labels.jpg" width="75%"/>
</p>

### Metrics curves

<p align="center">
<img src="assets/results.png" width="100%"/>
</p>

### Training results by class

```
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95):
                   all         11       1913      0.887      0.891       0.91      0.514
                 black         11       1261      0.897      0.873      0.891      0.339
                   key         11        114      0.998      0.991      0.995      0.602
                  line         11         27      0.964      0.995      0.991      0.792
                 piano         11          6      0.861          1      0.995      0.649
                  tone         11        204      0.672      0.608      0.682      0.275
                 voice         11         99      0.935       0.99      0.993      0.628
                 white         11        202      0.878      0.777      0.825      0.311
```


### Confusion matrix

<p align="center">
<img src="assets/confusion_matrix_normalized.png" width="75%"/>
</p>

### Ground Truth vs Prediction (validation dataset)

<p align="center">
<img src="assets/val_batch2_labels.jpg" width="40%"/> <img src="assets/val_batch2_pred.jpg" width="40%"/>
</p>

### Example of prediction

<p align="center">
<img src="assets/prediction.jpg" width="75%"/>
</p>



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


Update: Group solution

```python
from PIL import Image
import numpy as np
from shapely.geometry import box, Point
import os

def process_image_and_detections(image_path, detections_path):
    image = Image.open(image_path)
    image_width, image_height = image.size

    with open(detections_path, 'r') as file:
        lines = file.readlines()

    class_0_detections = [line.strip().split() for line in lines if line.startswith('0')]

    shapely_boxes = []
    centroids = []

    for detection in class_0_detections:
        _, x_center, y_center, width, height = map(float, detection)

        x_center = x_center * image_width
        width = width * image_width
        y_center = y_center * image_height
        height = height * image_height

        x_min = x_center - (width / 2)
        y_min = y_center - (height / 2)
        x_max = x_center + (width / 2)
        y_max = y_center + (height / 2)

        box_object = box(x_min, y_min, x_max, y_max)

        centroid = box_object.centroid

        shapely_boxes.append(box_object)
        centroids.append((centroid.x, centroid.y))

    sorted_boxes = [box for _, box in sorted(zip(centroids, shapely_boxes), key=lambda b: b[0][1])]

    for i, bbox in enumerate(sorted_boxes, 1):
        cropped_image = image.crop((bbox.bounds[0], bbox.bounds[1], bbox.bounds[2], bbox.bounds[3]))
        cropped_image.save(os.path.join(os.path.dirname(image_path), f"{i}_{os.path.basename(image_path)}"))

image_path = 'Downloads/ec94e196-Book02_00006.jpg'
detections_path = 'Downloads/ec94e196-Book02_00006.txt'

process_image_and_detections(image_path, detections_path)
```


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
