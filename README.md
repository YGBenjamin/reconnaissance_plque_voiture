# License Plate Recognition using YOLOv8 and EasyOCR

An end-to-end Computer Vision pipeline designed to detect vehicle license plates and extract their alphanumeric text. This project combines a custom-trained YOLOv8 object detection model with EasyOCR for Character Recognition, wrapped into a web interface.

## Web Application

The live application is deployed and accessible here:
👉 [https://ygbenjamin-rec-plaque.streamlit.app/](https://ygbenjamin-rec-plaque.streamlit.app/)

## Project Overview

The pipeline operates in two distinct phases:
1. **Object Detection (YOLOv8):** Locates and crops the license plate from an incoming image or video frame.
2. **Optical Character Recognition (EasyOCR):** Processes the cropped plate region to extract and clean the text.

## Dataset & Training

The model was trained locally on a custom dataset tailored for this specific detection task.

* **Data Annotation:** 200 images manually annotated using Label Studio.
* **Framework:** Ultralytics YOLOv8.
* **Training Depth:** Run over multiple epochs until convergence, monitoring box loss, class loss, and distribution focal loss (DFL) to ensure high localization accuracy.

## Performance & Metrics

The model achieved strong localization results. Based on the validation phase, here are the key performance metrics extracted from the final training logs:

| Metric | Value |
| :--- | :--- |
| **Precision (B)** | 95.9% |
| **Recall (B)** | 67.4% |
| **mAP50** | 76.5% |
| **mAP50-95** | 49.3% |

*Note: The high precision ensures that when a license plate is detected, it is almost always an actual license plate, minimizing false positives before passing the crop to the OCR engine.*

The repository includes the training evaluation plots (`results.jpg`, `confusion_matrix.png`, and precision-recall curves) for a detailed breakdown of the training history.

## Repository Structure

* `training.ipynb` - Jupyter Notebook containing the end-to-end YOLOv8 model training process.
* `preprocessing.py` - Python script handling data preparation, automated partitioning, and sorting into train, validation, and test splits.
* `detect_and_recognize.py` - Core processing script that runs the AI pipeline inference, extracts data from the model, and generates output videos.
* `app.py` - Streamlit application script powering the interactive web interface.
* `weights/` - Contains the trained custom YOLOv8 weights (e.g., `best.pt`, `last.pt`).
* `results.csv` - Raw training logs and epoch-by-epoch evaluation data.
* `curves/` - Precision, Recall, F1, and PR curves along with the normalized confusion matrix.

## Future Improvements

* Fine-tuning EasyOCR with a specific character whitelist (alphanumeric only) to avoid common misreadings (e.g., mistaking 'O' for '0' or 'I' for '1').
* Implementing a tracking algorithm (like SORT or DeepSORT) to stabilize text reading across sequential video frames.
