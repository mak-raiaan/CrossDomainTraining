
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO


# Project Paths
PROJECT_ROOT   = "/datasetfolderpath/anonymous/yolov11_training"

# Wild Dataset Base Path
WILD_BASE_PATH = f"{PROJECT_ROOT}/LoTE-Animal/wild"

# Web Dataset Base Path
WEB_BASE_PATH  = f"{PROJECT_ROOT}/LoTE-Animal/web"

TRAIN_IMAGE_DIR = f"{WILD_BASE_PATH}/train/images"
TRAIN_LABEL_DIR = f"{WILD_BASE_PATH}/train/labels"
VAL_IMAGE_DIR   = f"{WILD_BASE_PATH}/val/images"
VAL_LABEL_DIR   = f"{WILD_BASE_PATH}/val/labels"
TEST_IMAGE_DIR  = f"{WILD_BASE_PATH}/test/images"
TEST_LABEL_DIR  = f"{WILD_BASE_PATH}/test/labels"

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
    num_classes = 11  

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
        model = YOLO(RESUME_PATH)
        resume_flag = True
    else:
        print("\nNo checkpoint found. Starting fresh training!")
        model = YOLO("yolo11l.pt")
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
        optimizer='auto',
        lr0=0.001,
        weight_decay=0.0005,
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



# Validate

def validate(yaml_path, label):
    print(f"\nModel validation on {label} test set!")
    print("-" * 20)

    model = YOLO(BEST_MODEL_PATH)
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
    model = YOLO(BEST_MODEL_PATH)
    results = model(SAMPLE_IMAGE, save=True)
    results[0].show()


# Save Summary & Per-class CSV

def save_summary(test_metrics, num_classes):
    print("\n Training logs and summary")
    print("-" * 20)

    precision = float(np.mean(test_metrics.box.p))
    recall    = float(np.mean(test_metrics.box.r))

    print(" Final Training Results:")
    print(f" Model:          YOLOv11")
    print(f" Best model:     {BEST_MODEL_PATH}")
    print(f" Final mAP50:    {test_metrics.box.map50:.4f}")
    print(f" Final mAP50-95: {test_metrics.box.map:.4f}")
    print(f" Precision:      {precision:.4f}")
    print(f" Recall:         {recall:.4f}")

    class_performance = [
        (CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class_{i}",
         float(test_metrics.box.ap50[i]))
        for i in range(num_classes)
    ]
    sorted_performance = sorted(class_performance, key=lambda x: x[1], reverse=True)

    print("\n Top 5 Performing Classes:")
    for i, (class_name, ap50) in enumerate(sorted_performance[:5]):
        print(f"   {i+1}. {class_name}: {ap50:.4f}")

    print("\n Lowest 5 Performing Classes:")
    for i, (class_name, ap50) in enumerate(sorted_performance[-5:]):
        print(f"   {i+1}. {class_name}: {ap50:.4f}")

    # Save YAML summary
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_summary = {
        'model':            'YOLOv11',
        'dataset':          'LoTE-Animal Wild',
        'overall_mAP50':    float(test_metrics.box.map50),
        'overall_mAP':      float(test_metrics.box.map),
        'precision':        precision,
        'recall':           recall,
        'best_class':  sorted_performance[0][0],
        'worst_class': sorted_performance[-1][0],
    }
    summary_path = f"{OUTPUT_DIR}/yolov11_training_summary.yaml"
    with open(summary_path, 'w') as f:
        yaml.dump(results_summary, f)
    print(f"\n Summary saved: {summary_path}")

    # Save per-class CSV
    per_class_results = pd.DataFrame({
        'Class_ID':   range(num_classes),
        'Class_Name': [
            CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class_{i}"
            for i in range(num_classes)
        ],
        'AP50': test_metrics.box.ap50[:num_classes],
        'AP':   test_metrics.box.ap[:num_classes],
    })
    csv_path = f"{OUTPUT_DIR}/yolov11_per_class_results.csv"
    per_class_results.to_csv(csv_path, index=False)
    print(f" Per-class CSV saved: {csv_path}")



# Main

def main():
    # Verify dataset
    num_classes = verify_dataset()

    # Create YAMLs
    create_yaml(WILD_BASE_PATH, WILD_YAML_PATH, label="Wild")
    create_yaml(WEB_BASE_PATH,  WEB_YAML_PATH,  label="Web")

    # Train on wild
    train()

    # Validate — wild (same-domain)
    validate(WILD_YAML_PATH, label="Wild (same-domain)")

    # Sample inference
    run_inference()

    # Validate — web (cross-domain)
    test_metrics = validate(WEB_YAML_PATH, label="Web (cross-domain)")

    # Save summary using cross-domain metrics
    save_summary(test_metrics, num_classes)


if __name__ == "__main__":
    main()