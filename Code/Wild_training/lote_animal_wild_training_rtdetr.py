import os
import yaml
import numpy as np
import pandas as pd
from ultralytics import RTDETR


# Config

PROJECT_ROOT   = "/datasetfolderpath/anonymous/rtdetr_training"

# Wild Dataset Base Path
WILD_BASE_PATH = f"{PROJECT_ROOT}/LoTE-Animal/wild"

# Web Dataset Base Path
WEB_BASE_PATH  = f"{PROJECT_ROOT}/LoTE-Animal/web"

# Train/Val/Test Directories
TRAIN_IMAGE_DIR = f"{WILD_BASE_PATH}/train/images"
TRAIN_LABEL_DIR = f"{WILD_BASE_PATH}/train/labels"
VAL_IMAGE_DIR   = f"{WILD_BASE_PATH}/val/images"
VAL_LABEL_DIR   = f"{WILD_BASE_PATH}/val/labels"
TEST_IMAGE_DIR  = f"{WILD_BASE_PATH}/test/images"
TEST_LABEL_DIR  = f"{WILD_BASE_PATH}/test/labels"

# YAML Paths
WILD_YAML_PATH  = f"{PROJECT_ROOT}/working/wild_dataset.yaml"
WEB_YAML_PATH   = f"{PROJECT_ROOT}/working/web_dataset.yaml"
RESUME_PATH     = f"{PROJECT_ROOT}/working/wild/runs/detect/train/weights/last.pt"
BEST_MODEL_PATH = f"{PROJECT_ROOT}/working/wild/runs/detect/train/weights/best.pt"
OUTPUT_DIR      = f"{PROJECT_ROOT}/working/wild"
SAMPLE_IMAGE    = f"{PROJECT_ROOT}/LoTE-Animal/wild/test/images/000000000015.jpg"

CLASS_NAMES = [
    'Pandas', 'RedPandas', 'MartesFlavigula', 'MacacaThibetana',
    'RhinopithecusRoxellana', 'HystrixBrachyura', 'SusScrofa',
    'RusaUnicolor', 'ElaphodusCephalophus', 'CapricornisMilneedwardsii',
    'PseudoisNayaur'
]



# Verify Dataset Structure

def verify_dataset():
    print("\nVerifying dataset structure")
    print("Dataset Structure:")
    print("-" * 20)

    directories_to_check = [
        ('Train Images', TRAIN_IMAGE_DIR),
        ('Train Labels', TRAIN_LABEL_DIR),
        ('Val Images',   VAL_IMAGE_DIR),
        ('Val Labels',   VAL_LABEL_DIR),
        ('Test Images',  TEST_IMAGE_DIR),
        ('Test Labels',  TEST_LABEL_DIR),
    ]

    for name, dir_path in directories_to_check:
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path)
                     if f.endswith(('.jpg', '.png', '.jpeg', '.txt'))]
            file_type = "images" if 'image' in name.lower() else "labels"
            print(f"{name}: {len(files)} {file_type}")
            if files:
                print(f"   Sample: {files[0]}")
        else:
            print(f"{name}: Directory not found - {dir_path}")

    # Detect num_classes from labels
    print("\nChecking label format and classes...")
    num_classes = 11  # safe default

    if os.path.exists(TRAIN_LABEL_DIR):
        label_files = [f for f in os.listdir(TRAIN_LABEL_DIR) if f.endswith('.txt')]
        if label_files:
            sample_label_path = os.path.join(TRAIN_LABEL_DIR, label_files[0])
            with open(sample_label_path, 'r') as f:
                sample_content = f.readlines()
            if sample_content:
                print(f" Sample label format: {sample_content[0].strip()}")

            class_ids = set()
            for label_file in label_files[:20]:
                with open(os.path.join(TRAIN_LABEL_DIR, label_file), 'r') as f:
                    for line in f:
                        if line.strip():
                            try:
                                class_id = int(line.strip().split()[0])
                                class_ids.add(class_id)
                            except Exception:
                                continue

            print(f" Classes found in labels: {sorted(class_ids)}")
            num_classes = max(class_ids) + 1 if class_ids else 11
            print(f" Number of classes: {num_classes}")
        else:
            print("  No label files found, using default: 11 classes")
    else:
        print("  Labels directory not found, using default: 11 classes")

    return num_classes


# Create Dataset YAMLs

def create_yaml(base_path, yaml_path, label):
    print(f"\nCreating {label} dataset YAML configuration!")
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)

    content = f"""
path: {base_path}
train: train/images
val: val/images
test: test/images

nc: 11
names: {CLASS_NAMES}
"""
    with open(yaml_path, 'w') as f:
        f.write(content)

    print(f" Dataset YAML created: {yaml_path}")
    print("\nDataset YAML Content:")
    print("-" * 20)
    print(content)



# Train

def train():
    if os.path.exists(RESUME_PATH):
        print("\nCheckpoint found! Resuming training from last.pt")
        model = RTDETR(RESUME_PATH)
        resume_flag = True
    else:
        print("\nNo checkpoint found. Starting fresh training!")
        model = RTDETR("rtdetr-l.pt")
        resume_flag = False

    model.train(
        data=WILD_YAML_PATH,
        epochs=500,
        imgsz=800,
        batch=32,
        device=0,
        workers=4,
        patience=30,
        save=True,
        pretrained=not resume_flag,
        optimizer='AdamW',
        lr0=0.0001,
        weight_decay=0.0001,
        exist_ok=True,
        verbose=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.001,
        flipud=0.01,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.1,
        copy_paste=0.1,
        resume=resume_flag,
    )

    print("\nTraining Complete!")
    if resume_flag:
        print("Training continued from previous checkpoint.")
    else:
        print("Training started fresh.")


# Validate on Test Set

def validate(yaml_path, label):
    print(f"\nModel validation on {label} test set!")
    print("-" * 20)

    model = RTDETR(BEST_MODEL_PATH)
    test_metrics = model.val(
        data=yaml_path,
        split='test',
        imgsz=800,
    )

    print(f"\nValidation Results ({label}):")
    print(f" mAP50:     {test_metrics.box.map50:.4f}")
    print(f" mAP50-95:  {test_metrics.box.map:.4f}")
    print(f" Precision: {test_metrics.box.p.mean():.4f}")
    print(f" Recall:    {test_metrics.box.r.mean():.4f}")

    return test_metrics

# Sample Inference

def run_inference():
    print("\nRunning sample inference...")
    model = RTDETR(BEST_MODEL_PATH)
    results = model(SAMPLE_IMAGE, save=True)
    results[0].show()

# Main

def main():
    # Verify dataset
    num_classes = verify_dataset()

    # Create YAMLs
    create_yaml(WILD_BASE_PATH, WILD_YAML_PATH, label="Wild")
    create_yaml(WEB_BASE_PATH,  WEB_YAML_PATH,  label="Web")

    # Train on wild
    train()

    # Validate on wild (same-domain)
    validate(WILD_YAML_PATH, label="Wild (same-domain)")

    # Step 5: Sample inference
    run_inference()

    # Validate on web (cross-domain)
    test_metrics = validate(WEB_YAML_PATH, label="Web (cross-domain)")


if __name__ == "__main__":
    main()
