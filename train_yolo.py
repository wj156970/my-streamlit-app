import wget
import os

# 下载YOLOv5预训练权重
url = 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt'
if not os.path.exists('yolov5s.pt'):
    print("下载YOLOv5预训练权重...")
    wget.download(url, 'yolov5s.pt')
    print("下载完成!")
else:
    print("预训练权重已存在!")

# 训练模型
import subprocess
import sys

cmd = [
    sys.executable, '-m', 'yolov5.train',
    '--img', '640',
    '--batch', '16',
    '--epochs', '50',
    '--data', 'banana_dataset.yaml',
    '--weights', 'yolov5s.pt',
    '--name', 'banana_detection_model',
    '--cache'
]

print("开始训练模型...")
subprocess.run(cmd, check=True)
print("训练完成!")