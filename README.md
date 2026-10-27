# Closing the Web-Wild Domain Gap: Structured Mixed-Domain Training for Robust Camera-Trap Wildlife Detection

Official implementation of our research on **cross-domain wildlife object detection**, investigating the performance gap between **Web** and **Wild** domains using **YOLOv11**, **YOLO26**, and **RT-DETR** on the LoTE-Animal dataset.

---

## Overview

Wildlife object detectors often perform well when trained and evaluated within the same domain but experience significant performance degradation when transferred across domains. This repository systematically benchmarks this **Web-to-Wild** and **Wild-to-Web** domain gap using three state-of-the-art object detection models.

The repository includes:

* Wild-domain training
* Web-domain training
* Mixed-domain training
* Same-domain evaluation
* Cross-domain evaluation
* Reproducible experimental settings

---

## Repository

GitHub Repository:

**[Web-to-Wild-Wildlife-Detection](https://github.com/nomanhossainn/Web-to-Wild-Wildlife-Detection.git)**

Clone the repository:

```bash
git clone https://github.com/nomanhossainn/Web-to-Wild-Wildlife-Detection.git
cd Web-to-Wild-Wildlife-Detection
```

---

# Repository Structure

```
Web-to-Wild-Wildlife-Detection/Code/

├── wild_training/
│   ├── lote_animal_wild_training_yolov11.py
│   ├── lote_animal_wild_training_yolov26.py
│   └── lote_animal_wild_training_rtdetr.py
│
├── web_training/
│   ├── lote-animal-web-training_yolov11.ipynb
│   ├── lote-animal-web-training_yolo26.ipynb
│   └── lote-animal-web-training_rtdetr.ipynb
│
└── Mixed_data_training/
    ├── Setup1/   (training on 80% Wild + 20% Web, tested on 100% Wild)
    │   ├── mixed-domain-training-S1-80wild-20web_rtdetr.ipynb
    │   ├── mixed-domain-training-S1-80wild-20web_yolo26.ipynb
    │   └── mixed-domain-training-S1-80wild-20web_yolov11.ipynb
    │
    ├── Setup2/   (training on 60% Wild + 40% Web, tested on unused 40% Wild and 60% Web)
    │   ├── lote-animal-mix-training_S2_rtdetr.ipynb
    │   ├── lote-animal-mix-training_S2_yolo26.ipynb
    │   └── lote-animal-mix-training_S2_yolov11.ipynb
    │
    ├── Setup3/   (80% Web + 20% Wild, tested on 100% Web)
    │   ├── mixed-domain-S3-80web-20wild-training_rtdetr.ipynb
    │   ├── mixed-domain-S3-80web-20wild-training_yolo26.ipynb
    │   └── mixed-domain-S3-80web-20wild-training_yolov11.ipynb
    │
    └── Setup4/   (60% Web + 40% Wild, Unused 60% Wild and 40% Web)
        ├── lote-animal-mix-training_S4_rtdetr.ipynb
        ├── lote-animal-mix-training_S4_yolo26.ipynb
        └── lote-animal-mix-training_S4_yolov11.ipynb
```

---

# Dataset

This work uses the **LoTE-Animal** dataset.

Official Website

**[LoTE-Animal Dataset](https://lote-animal.github.io/)**

Kaggle Download

**[LoTE-Animal on Kaggle](https://www.kaggle.com/datasets/mdnomanhossain/lote-animal)**

---

# Download using KaggleHub

```python
import kagglehub

path = kagglehub.dataset_download("mdnomanhossain/lote-animal")

print("Path to dataset files:", path)
```

---

# Dataset Structure

```
LoTE-Animal/

├── web/
│   ├── train/
│   ├── val/
│   └── test/
│
└── wild/
    ├── train/
    ├── val/
    └── test/
```

Each split contains

```
images/
labels/
```

using YOLO annotation format.

---

# Classes

The dataset contains **11 wildlife categories**.

* Pandas
* RedPandas
* MartesFlavigula
* MacacaThibetana
* RhinopithecusRoxellana
* HystrixBrachyura
* SusScrofa
* RusaUnicolor
* ElaphodusCephalophus
* CapricornisMilneedwardsii
* PseudoisNayaur

---

# Installation

Install the required packages.

```bash
pip install ultralytics
pip install torch torchvision
pip install numpy
pip install pandas
pip install pyyaml
pip install kagglehub
```

---

# Configure Dataset Path

Open the training script and modify

```python
PROJECT_ROOT="/datasetfolderpath/anonymous"
```

to your local dataset location.

Example

```python
PROJECT_ROOT="/home/user/LoTE-Animal"
```

or

```python
PROJECT_ROOT="D:/datasets/LoTE-Animal"
```

---

# Wild Training

Folder:

```
wild_training/
```

### lote_animal_wild_training_yolov11.py

* Trains YOLOv11 using the Wild dataset
* Evaluates on Wild test set
* Evaluates on Web test set
* Performs sample inference

### lote_animal_wild_training_yolo26.py

Same training pipeline using YOLO26.

### lote_animal_wild_training_rtdetr.py

Same training pipeline using RT-DETR.

The script automatically

* verifies dataset structure
* checks annotation files
* creates dataset YAML files
* resumes training from the checkpoint if available
* trains the model
* validates on the same domain
* validates on the cross-domain
* performs sample inference

---

# Web Training

Folder:

```
web_training/
```

Contains notebooks for

* YOLOv11
* YOLO26
* RT-DETR

Each notebook

* trains on Web images
* evaluates on Web test set
* evaluates on Wild test set
* performs inference

---

## Mixed-Domain Training Setups

To investigate the impact of different domain compositions on wildlife object detection, four mixed-domain training strategies were designed. Each setup combines different proportions of **Web** and **Wild** images while evaluating the models on the corresponding test set.

| **Setup** | **Training Data**  | **Testing Data**                       | **Objective**                                                                                       |
| :-------: | ------------------ | -------------------------------------- | --------------------------------------------------------------------------------------------------- |
|   **S1**  | 80% Wild + 20% Web | 100% Wild                              | Evaluate whether a small amount of Web images can improve detection performance on the Wild domain. |
|   **S2**  | 60% Wild + 40% Web | Remaining 40% Wild + Remaining 60% Web | Assess balanced mixed-domain training using unseen images from both domains.                        |
|   **S3**  | 80% Web + 20% Wild | 100% Web                               | Investigate whether a small amount of Wild images improves detection performance on the Web domain. |
|   **S4**  | 60% Web + 40% Wild | Remaining 40% Web + Remaining 60% Wild | Evaluate a Web-dominant mixed-domain training strategy using unseen images from both domains.       |

### Mixed-Domain Training Files

| **Setup**   | **YOLOv11**                                           | **YOLO26**                                           | **RT-DETR**                                          |
| ----------- | ----------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------- |
| **Setup 1** | `mixed-domain-training-S1-80wild-20web_yolov11.ipynb` | `mixed-domain-training-S1-80wild-20web_yolo26.ipynb` | `mixed-domain-training-S1-80wild-20web_rtdetr.ipynb` |
| **Setup 2** | `lote-animal-mix-training_S2_yolov11.ipynb`           | `lote-animal-mix-training_S2_yolo26.ipynb`           | `lote-animal-mix-training_S2_rtdetr.ipynb`           |
| **Setup 3** | `mixed-domain-S3-80web-20wild-training_yolov11.ipynb` | `mixed-domain-S3-80web-20wild-training_yolo26.ipynb` | `mixed-domain-S3-80web-20wild-training_rtdetr.ipynb` |
| **Setup 4** | `lote-animal-mix-training_S4_yolov11.ipynb`           | `lote-animal-mix-training_S4_yolo26.ipynb`           | `lote-animal-mix-training_S4_rtdetr.ipynb`           |


# Running the Experiments

YOLOv11

```bash
python lote_animal_wild_training_yolov11.py
```

YOLO26

```bash
python lote_animal_wild_training_yolo26.py
```

RT-DETR

```bash
python lote_animal_wild_training_rtdetr.py
```

For Web and Mixed-domain experiments, open the corresponding Jupyter notebook and execute all cells.

---

# Training Pipeline

Each training script follows the same workflow:

1. Verify dataset structure
2. Validate annotations
3. Generate dataset YAML files
4. Load pretrained model
5. Resume training if a checkpoint exists
6. Train the detector
7. Evaluate on the same-domain test set
8. Evaluate on the cross-domain test set
9. Run sample inference

---

# Evaluation Metrics

The following metrics are reported:

* mAP@50
* mAP@50–95
* Precision
* Recall

---

# Output Files

Training results are automatically saved by Ultralytics.

Typical outputs include:

```
runs/
weights/
best.pt
last.pt
results.csv
results.png
confusion_matrix.png
PR_curve.png
```

---

# Reproducing the Experiments

1. Clone the repository.
2. Download the LoTE-Animal dataset.
3. Update the `PROJECT_ROOT` path.
4. Install the required packages.
5. Run the desired training script or notebook.
6. Compare the generated metrics with those reported in the paper.

---

# Notes

* All annotations use YOLO format.
* Dataset YAML files are generated automatically.
* Training resumes automatically if a previous checkpoint is detected.
* Hyperparameters can be modified directly within each training script.


---

# Acknowledgements

We thank the developers of:

* LoTE-Animal Dataset
* Ultralytics
* RT-DETR

for making their datasets and frameworks publicly available.


# Contact

If you have questions or encounter issues, please open a GitHub Issue in this repository.

Happy researching! 🚀
