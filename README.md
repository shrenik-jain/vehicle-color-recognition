Color classifying by K-Nearest Neighbors Machine Learning Classifier which is trained by R, G, B Color Histogram. It can classify White, Black, Red, Green, Blue, Orange, Yellow, and Violet.

### Introduction
This repository contains the code for Car Detection and Color Classification.

---

### Repository Structure

```
├── color_classification.py
├── color_recognition_api
│   ├── color_histogram_feature_extraction.py
│   ├── __init__.py
│   ├── knn_classifier.py
├── Haarcascades
│   └── haarcascade_car.xml
├── README.md
├── requirements.txt
├── sample_videos
│   ├── car_driving.mp4
│   └── car.png
├── test.data
├── training.data
└── training_dataset
    ├── black
    ├── blue
    ├── green
    ├── orange
    ├── red
    ├── violet
    ├── white
    └── yellow
```

---

### Requirements and Implementation
- All the requirements can be installed by running the command `pip3 install -r requirements.txt`

- You can add a video of your choice in [sample_videos]() or use the default video

- Running the command `python3 color_classification.py` would execute the code

---

### References

- [color_recognition_api](https://github.com/ahmetozlu/color_recognition/tree/master/src/color_recognition_api) -> For Real-Time color Detection