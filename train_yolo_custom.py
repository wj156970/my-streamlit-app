import torch
import os
import sys
import yaml
import subprocess
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

# 获取YOLOv5的train.py路径
yolov5_path = os.path.dirname(yolov5.__file__)
train_script = os.path.join(yolov5_path, 'train.py')

print(f"\n使用训练脚本: {train_script}")

# 尝试创建一个简单的模型配置
model_cfg = """
# YOLOv5s configuration for banana detection

# Parameters
nc: 4  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5s backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Focus, [64, 3]],  # 1-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 2-P2/4
   [-1, 3, C3, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 4-P3/8
   [-1, 9, C3, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 6-P4/16
   [-1, 9, C3, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 8-P5/32
   [-1, 1, SPPF, [1024, 5]]]  # 9

# YOLOv5s head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C3, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3, [256, False]],  # 17 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)

   [[17, 20, 23], 1, Detect, [nc, anchors]]]  # Detect(P3, P4, P5)
"""

# 保存模型配置
with open('banana_yolov5s.yaml', 'w') as f:
    f.write(model_cfg)

print("\n创建自定义模型配置文件: banana_yolov5s.yaml")

# 创建训练命令
cmd = [
    sys.executable, train_script,
    '--img', '320',  # 更小的图像尺寸
    '--batch', '4',  # 更小的批处理大小
    '--epochs', '5',  # 更少的训练轮数
    '--data', 'banana_dataset.yaml',
    '--cfg', 'banana_yolov5s.yaml',  # 使用自定义配置
    '--weights', '',  # 从零开始
    '--name', 'banana_detection_model_custom',
    '--device', str(device),
    '--exist-ok',
    '--nosave'  # 不保存中间结果，只保存最终模型
]

print("\n执行命令:", ' '.join(cmd))

# 执行训练
try:
    result = subprocess.run(cmd, check=True)
    print("\n训练完成!")
except subprocess.CalledProcessError as e:
    print(f"\n训练失败，错误代码: {e.returncode}")
    
    # 尝试使用预训练权重
    print("\n尝试使用预训练权重...")
    cmd_with_weights = [
        sys.executable, train_script,
        '--img', '320',
        '--batch', '4',
        '--epochs', '5',
        '--data', 'banana_dataset.yaml',
        '--cfg', 'banana_yolov5s.yaml',
        '--weights', 'yolov5s.pt',  # 使用预训练权重
        '--name', 'banana_detection_model_pretrained',
        '--device', str(device),
        '--exist-ok',
        '--nosave'
    ]
    
    print("执行命令:", ' '.join(cmd_with_weights))
    try:
        result = subprocess.run(cmd_with_weights, check=True)
        print("\n使用预训练权重的训练完成!")
    except subprocess.CalledProcessError as e2:
        print(f"\n使用预训练权重的训练也失败，错误代码: {e2.returncode}")
        print("请检查数据集配置和YOLOv5安装")