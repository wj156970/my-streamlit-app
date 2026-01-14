# Banana Ripeness Detection Training Script
# This is a placeholder script that will use YOLOv5 for training
# Once YOLOv5 is available, this script can be updated for actual training

import os
import torch
import yaml

# Print system information
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Load dataset configuration
with open('banana_dataset.yaml', 'r') as f:
    dataset_config = yaml.safe_load(f)

print("\nDataset Configuration:")
print(f"Dataset path: {dataset_config['path']}")
print(f"Classes: {dataset_config['names']}")
print(f"Number of classes: {dataset_config['nc']}")

# Dataset structure validation
print("\nValidating dataset structure...")

for split in ['train', 'val', 'test']:
    split_path = os.path.join(dataset_config['path'], dataset_config[split])
    if os.path.exists(split_path):
        print(f"✓ {split} directory exists")
        for class_name in dataset_config['names']:
            class_path = os.path.join(split_path, class_name)
            if os.path.exists(class_path):
                num_images = len(os.listdir(class_path))
                print(f"  - {class_name}: {num_images} images")
            else:
                print(f"  ✗ {class_name} directory missing")
    else:
        print(f"✗ {split} directory missing")

print("\nDataset validation complete.")
print("\nTo train the model, please install YOLOv5 and run:")
print("python yolov5/train.py --data banana_dataset.yaml --cfg yolov5s.yaml --weights '' --epochs 100 --batch-size 16 --name banana_ripeness")
