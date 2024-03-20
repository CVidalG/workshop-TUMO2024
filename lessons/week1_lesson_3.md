# Lesson 3 (week 1)

Objectives:

* Object detection labeling
* Coding a data split function
* Training an object dectection model and evaluating it
* Training a classification model

## The Komitas Dataset
[Download link](https://drive.google.com/drive/folders/1T01oMvwHWQ0y3VtXUr3DARhs-LJZwnyJ?usp=sharing)

### Annotation Guidelines

<p align="center">
<img src="assets/object_labeling.png" width="100%"/>
</p>

List of training classes:
* `voice`
* `line`
* `piano`
* `white`
* `black`
* `key`
* `tone`

# Object detection training mode of YOLO

```python
model = YOLO('yolov8n')
results = model.train() #with params of your choices
```

## Datasets for Image Classification using YOLO

```python
model = YOLO('yolov8n-cls') # classification
results = model.train() # with params of your choices
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

