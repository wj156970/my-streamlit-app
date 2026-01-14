import torch
import os
import sys
import yaml
import cv2
import numpy as np
from pathlib import Path

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 读取数据集配置
with open('banana_dataset.yaml', 'r') as f:
    data_config = yaml.safe_load(f)

print(f"数据集路径: {data_config['path']}")
print(f"类别数量: {data_config['nc']}")
print(f"类别名称: {data_config['names']}")

# 检查数据集路径是否存在
dataset_path = Path(data_config['path'])
if not dataset_path.exists():
    print(f"错误: 数据集路径不存在: {dataset_path}")
    sys.exit(1)

# 检查训练和验证目录
train_path = dataset_path / data_config['train']
val_path = dataset_path / data_config['val']

if not train_path.exists():
    print(f"错误: 训练目录不存在: {train_path}")
    sys.exit(1)

if not val_path.exists():
    print(f"错误: 验证目录不存在: {val_path}")
    sys.exit(1)

# 统计数据集中的图像数量
def count_images_and_labels(path):
    image_count = 0
    label_count = 0
    for class_dir in path.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob('*.jpg'))
            labels = list(class_dir.glob('*.txt'))
            image_count += len(images)
            label_count += len(labels)
            print(f"  {class_dir.name}: {len(images)} 图像, {len(labels)} 标签")
    return image_count, label_count

print("\n训练集统计:")
train_images, train_labels = count_images_and_labels(train_path)

print("\n验证集统计:")
val_images, val_labels = count_images_and_labels(val_path)

print(f"\n总计: {train_images} 训练图像, {val_images} 验证图像")

# 检查YOLOv5是否安装
try:
    import yolov5
    print(f"\nYOLOv5版本: {yolov5.__version__}")
except ImportError:
    print("\n错误: YOLOv5未安装")
    sys.exit(1)

# 尝试创建一个简单的模型进行训练
print("\n准备训练模型...")

# 由于网络问题，我们将使用一个简化的方法
# 首先尝试从本地加载预训练权重
weights_path = 'yolov5s.pt'
if not os.path.exists(weights_path):
    print(f"警告: 预训练权重文件不存在: {weights_path}")
    print("将尝试从零开始训练...")
    weights_path = ''

# 创建训练命令
cmd = [
    sys.executable, '-m', 'yolov5.train',
    '--img', '640',
    '--batch', '8',  # 减少批处理大小以避免内存问题
    '--epochs', '10',  # 减少训练轮数以加快速度
    '--data', 'banana_dataset.yaml',
    '--weights', weights_path if weights_path else '',
    '--name', 'banana_detection_model',
    '--cache', 'ram',  # 使用RAM缓存以加快训练速度
    '--device', str(device),
    '--exist-ok'  # 允许覆盖现有模型
]

print("执行命令:", ' '.join(cmd))

# 执行训练
import subprocess
try:
    result = subprocess.run(cmd, check=True)
    print("\n训练完成!")
except subprocess.CalledProcessError as e:
    print(f"\n训练失败，错误代码: {e.returncode}")
    print("尝试使用更简单的配置重新训练...")
    
    # 尝试使用更简单的配置
    simple_cmd = [
        sys.executable, '-m', 'yolov5.train',
        '--img', '320',  # 更小的图像尺寸
        '--batch', '4',  # 更小的批处理大小
        '--epochs', '5',  # 更少的训练轮数
        '--data', 'banana_dataset.yaml',
        '--weights', '',  # 从零开始
        '--name', 'banana_detection_model_simple',
        '--device', str(device),
        '--exist-ok'
    ]
    
    print("执行简化命令:", ' '.join(simple_cmd))
    try:
        result = subprocess.run(simple_cmd, check=True)
        print("\n简化训练完成!")
    except subprocess.CalledProcessError as e2:
        print(f"\n简化训练也失败，错误代码: {e2.returncode}")
        print("请检查数据集配置和YOLOv5安装")