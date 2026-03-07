# Solar Panel Health AI

## Overview

Solar Panel Health AI is a machine learning research project focused on **automatic fault detection and predictive health analysis of photovoltaic (PV) panels using thermal images**.

The goal is to move beyond simple fault classification and build a **predictive maintenance system** that can:

* Detect photovoltaic faults from thermal images
* Estimate the current health of a solar panel
* Predict future degradation of the panel
* Provide maintenance recommendations

This project combines **computer vision, machine learning, and photovoltaic degradation modeling**.

---

# Dataset

This project uses the **Infrared Solar Modules Dataset**.

Dataset characteristics:

* Total images: **20,000**
* Image type: **Thermal infrared**
* Total classes: **12 fault categories**

### Fault Classes

* No-Anomaly
* Cell
* Cell-Multi
* Shadowing
* Cracking
* Diode
* Diode-Multi
* Hot-Spot
* Hot-Spot-Multi
* Vegetation
* Soiling
* Offline-Module

### Class Distribution

| Class          | Count |
| -------------- | ----- |
| No-Anomaly     | 10000 |
| Cell           | 1877  |
| Vegetation     | 1639  |
| Diode          | 1499  |
| Cell-Multi     | 1288  |
| Shadowing      | 1056  |
| Cracking       | 940   |
| Offline-Module | 827   |
| Hot-Spot       | 249   |
| Hot-Spot-Multi | 246   |
| Soiling        | 204   |
| Diode-Multi    | 175   |

The dataset shows **significant class imbalance**, which will be handled later during model training.

---

# Project Structure

```
solar-panel-health-ai
│
├── data
│   ├── raw
│   │   ├── images
│   │   └── module_metadata.json
│   │
│   ├── processed
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   │
│   └── split_dataset.py
│
├── src
│   ├── data
│   │   └── explore_dataset.py
│
├── notebooks
│
├── app
│
├── output
│
├── requirements.txt
```

---

# Current Progress

## Dataset Exploration

Script:

```
src/data/explore_dataset.py
```

This script:

* Loads dataset metadata from JSON
* Converts metadata into a structured DataFrame
* Calculates class distribution
* Displays dataset statistics

Example output:

```
Dataset size: 20000
```

---

## Dataset Splitting

Script:

```
data/split_dataset.py
```

The dataset is split into:

* **Train: 70% (14,000 images)**
* **Validation: 15% (3,000 images)**
* **Test: 15% (3,000 images)**

Generated files:

```
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
```

Each CSV contains:

```
image_path,label
images/13357.jpg,No-Anomaly
images/19719.jpg,Hot-Spot
```

---

# Planned System Architecture

The final system will follow this pipeline:

```
Thermal Image
      ↓
CNN Fault Classification
      ↓
Thermal Feature Extraction
      ↓
Panel Health Score Model
      ↓
Degradation Prediction Model
      ↓
Maintenance Recommendation
```

---

# Future Components

The next stages of the project include:

### 1. PyTorch Dataset Pipeline

Load images and labels from CSV and convert them into tensors.

### 2. CNN Fault Classification

Train a deep learning model (ResNet50 / EfficientNet) to classify panel faults.

### 3. Thermal Feature Extraction

Extract features such as:

* hotspot count
* hotspot area
* temperature variance
* maximum temperature

### 4. Panel Health Score Model

Estimate panel health on a **0–100 scale** using regression models.

### 5. Degradation Prediction Model

Predict **future panel health** using degradation rates from photovoltaic research literature.

### 6. Maintenance Recommendation System

Generate actionable maintenance suggestions.

### 7. Web Interface

Build a simple interface using **Streamlit** to upload thermal images and view predictions.

---

# Research Goal

Most solar panel AI systems only perform **fault classification**.

This project aims to build a **predictive solar panel monitoring system** that performs:

* Fault detection
* Health estimation
* Future degradation prediction

This enables **predictive maintenance for photovoltaic systems**.

---

# Technologies Used

* Python
* PyTorch
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Streamlit (planned)

---

# Project Status

Current stage:

```
Dataset exploration ✔
Dataset split ✔
Model development ⏳
```

---

# Author

Akshat Singh
B.Tech Computer Science
