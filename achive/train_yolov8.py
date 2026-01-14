import torch
import os
import sys
import yaml
from pathlib import Path

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 读取数据集配置
with open('../banana_dataset.yaml', 'r') as f:
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

# 检查YOLOv8是否安装
try:
    from ultralytics import YOLO
    print("\n成功导入YOLOv8")
except ImportError:
    print("\n错误: YOLOv8未安装")
    sys.exit(1)

# 创建YOLOv8模型
print("\n创建YOLOv8模型...")
model = YOLO('../yolov8n.pt')  # 使用nano版本，适合快速训练

# 训练模型
print("\n开始训练模型...")
results = model.train(
    data='banana_dataset.yaml',
    epochs=10,  # 训练轮数
    imgsz=320,  # 图像尺寸
    batch=8,    # 批处理大小
    name='banana_detection_yolov8',  # 实验名称
    device=device,  # 使用GPU或CPU
    exist_ok=True,  # 允许覆盖现有实验
    verbose=True  # 显示详细输出
)

print("\n训练完成!")
print(f"模型保存位置: {results.save_dir}")

# 验证模型
print("\n验证模型性能...")
metrics = model.val()

print("\n验证指标:")
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")

# 保存最终模型
final_model_path = '../banana_detection_yolov8_final.pt'
model.save(final_model_path)
print(f"\n最终模型已保存至: {final_model_path}")