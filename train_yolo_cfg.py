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

# 获取YOLOv5的默认配置文件路径
yolov5_path = os.path.dirname(yolov5.__file__)
cfg_path = os.path.join(yolov5_path, 'models', 'yolov5s.yaml')
hyp_path = os.path.join(yolov5_path, 'data', 'hyps', 'hyp.scratch-low.yaml')

print(f"使用配置文件: {cfg_path}")
print(f"使用超参数文件: {hyp_path}")

# 创建训练命令
cmd = [
    sys.executable, '-m', 'yolov5.train',
    '--img', '320',  # 更小的图像尺寸
    '--batch', '4',  # 更小的批处理大小
    '--epochs', '5',  # 更少的训练轮数
    '--data', 'banana_dataset.yaml',
    '--cfg', cfg_path,  # 使用配置文件
    '--weights', '',  # 从零开始
    '--hyp', hyp_path,  # 使用超参数文件
    '--name', 'banana_detection_model_simple',
    '--device', str(device),
    '--exist-ok'
]

print("执行命令:", ' '.join(cmd))

# 执行训练
import subprocess
try:
    result = subprocess.run(cmd, check=True)
    print("\n训练完成!")
except subprocess.CalledProcessError as e:
    print(f"\n训练失败，错误代码: {e.returncode}")
    print("请检查数据集配置和YOLOv5安装")