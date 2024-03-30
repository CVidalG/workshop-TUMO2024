# Lesson 2 (week 1)

Objectives:

* Understanding differences and requirements for several Computer Vision tasks
* Learning to define guidelines for Object detection labeling
* Familiarizing yourself with Label Studio (for Computer Vision)
* Training a classification model

## Computer Vision tasks

<p align="center">
<img src="assets/sheeps_TDS.jpg" width="100%"/>
</p>

[source image](https://towardsdatascience.com/detection-and-segmentation-through-convnets-47aa42de27ea)


## Computer Vision with YOLO V8 (You Only Look Once)

*Official documentation* : [https://docs.ultralytics.com](https://docs.ultralytics.com)

* Tasks: classify, detect, segment, pose
* Model: Train, Predict

## Installing Label Studio

[Label Studio](https://labelstud.io) is a very versatile and user-friendly labeling tool, with various templates for multiple ML tasks (audio, text, image, etc.).

<p align="center">
<img src="assets/labelstudio.png" width="100%"/>
</p>

1. Create an environment named `labelstudio` with the following command:

```bash
conda create --name labelstudio python=3.11
```

2. Activate your `labelstudio` environment.

3. Install python dependencies needed for the workshop:
```bash
pip install label-studio
```

4. Launching Label Studio:
```bash
label-studio
```