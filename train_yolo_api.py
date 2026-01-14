import torch
import os
import sys
import yaml
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

# 尝试使用YOLOv5的API进行训练
print("\n准备使用YOLOv5 API训练模型...")

try:
    # 尝试加载预训练模型
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    print("成功加载预训练模型")
    
    # 尝试训练模型
    try:
        results = model.train(data='banana_dataset.yaml', epochs=5, imgsz=320, batch=4)
        print("\n训练完成!")
        print(f"模型保存位置: {results.save}")
    except Exception as e:
        print(f"\n使用API训练失败: {str(e)}")
        
        # 尝试使用本地文件
        print("\n尝试使用本地文件训练...")
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', source='local')
        results = model.train(data='banana_dataset.yaml', epochs=5, imgsz=320, batch=4)
        print("\n训练完成!")
        print(f"模型保存位置: {results.save}")
        
except Exception as e:
    print(f"\n加载预训练模型失败: {str(e)}")
    print("尝试从零开始训练...")
    
    try:
        # 从零开始训练
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False)
        results = model.train(data='banana_dataset.yaml', epochs=5, imgsz=320, batch=4)
        print("\n训练完成!")
        print(f"模型保存位置: {results.save}")
    except Exception as e2:
        print(f"\n从零开始训练也失败: {str(e2)}")
        print("请检查网络连接和YOLOv5安装")