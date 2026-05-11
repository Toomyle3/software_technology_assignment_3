# Macroinvertebrate Image Analysis System

A small Python application that analyses freshwater macroinvertebrate images.
It scans an image dataset, runs exploratory data analysis (EDA), trains a
baseline image classifier, and offers a desktop GUI and a menu-driven
console interface for predicting the class of a new image.

This project was built for the unit **Software Technology 1 (4483 / 8995)**,
Assignment 3.

## Project Goal

- index a folder of macroinvertebrate images by class
- explore the dataset with charts and summary statistics
- train a baseline classifier (Random Forest on flattened grayscale features)
- deploy the trained model through a Tkinter GUI or a console menu

## Main Features

- dataset indexing (each sub-folder becomes a class label)
- class imbalance, image-size, aspect-ratio, and pixel-intensity analyses
- sample-image grid per class
- noisy / inconsistent image detection (too dark, too bright, unusual size)
- baseline classification with training/test split and saved evaluation report
- Tkinter desktop app to predict the class of a chosen image
- menu-driven console app with the same actions

## Packages Used

- `pandas`, `numpy` — data tables and numeric arrays
- `opencv-python` — read, resize and preprocess images
- `matplotlib`, `seaborn` — EDA charts and confusion matrix
- `scikit-learn` — train/test split, Random Forest, evaluation metrics
- `joblib` — save and load the trained model
- `Pillow` — display images inside the Tkinter window
- `tkinter` — Stage 3 desktop GUI (part of the Python standard library)

## Folder Structure

```text
software_technology_assignment_3/
├── data/
│   └── raw/macro_small/         # one sub-folder per class
├── outputs/
│   ├── eda/                     # EDA charts and CSV summaries
│   ├── models/                  # saved classifier (.joblib)
│   └── reports/                 # classification report + confusion matrix
├── src/
│   ├── config.py                # paths and shared settings
│   ├── main.py                  # batch run (Stage 1 + Stage 2)
│   ├── app.py                   # Tkinter desktop app entry
│   ├── console_app.py           # menu-driven console entry
│   ├── models/
│   │   └── records.py           # ImageRecord dataclass
│   └── services/
│       ├── dataset_indexer.py
│       ├── eda_service.py
│       ├── image_preprocessor.py
│       ├── classifier_service.py
│       ├── workflow_service.py  # ties every service together
│       ├── gui_app.py           # Tkinter window class
│       └── console_app.py       # console menu class
├── MANUAL_TESTING.md
├── requirements.txt
└── README.md
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the dataset under `data/raw/macro_small/<class_name>/*.png`.

## How to Run

From inside the `src/` folder:

```bash
# Stage 1 + Stage 2 in one go (indexes data, runs EDA, trains the model)
python main.py

# Stage 3: Tkinter desktop app (load an image and predict)
python app.py

# Optional: menu-driven console version
python console_app.py
```

Outputs land in `outputs/eda/`, `outputs/models/` and `outputs/reports/`.

## Class Overview

| Class | Responsibility |
| --- | --- |
| `ImageRecord` | Holds the metadata for one indexed image (data class). |
| `DatasetIndexer` | Walks the dataset folder and builds the image table. |
| `ImagePreprocessor` | Converts a raw image into flat numeric features for the model. |
| `EDAService` | Generates EDA charts, summary statistics, and noisy-image flags. |
| `ClassifierService` | Trains, evaluates, saves, and reuses the baseline classifier. |
| `WorkflowService` | Coordinates the shared steps used by every entry script. |
| `MacroinvertebrateApp` | Tkinter desktop application for picking and predicting images. |
| `ConsoleApp` | Menu-driven console application with the same actions. |

## Acknowledgement of Reused or Adapted Code

Parts of the code structure (folder layout, `WorkflowService` pattern,
baseline classifier outline) were adapted from the Week 10 and Week 11
Assignment 3 guidance materials. The transfer-learning option from the
guidance was not used; we kept the simpler OpenCV + Scikit-learn baseline.
