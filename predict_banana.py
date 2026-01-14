import torch
import cv2
import os
from pathlib import Path
from PIL import Image
import numpy as np

def predict_banana_ripeness(image_path, model_path='banana_detection_yolov8_final.pt'):
    """
    使用训练好的YOLOv8模型预测香蕉的成熟度
    
    参数:
        image_path: 输入图像路径
        model_path: 训练好的模型路径
    
    返回:
        检测结果和类别标签
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("错误: 未安装ultralytics库")
        return None
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return None
    
    # 检查图像文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 图像文件不存在: {image_path}")
        return None
    
    # 加载模型
    model = YOLO(model_path)
    
    # 进行预测
    results = model.predict(image_path)
    
    # 类别标签
    class_names = ['overripe', 'ripe', 'rotten', 'unripe']
    
    # 处理结果
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                # 获取置信度
                conf = box.conf[0].cpu().numpy()
                # 获取类别
                cls = int(box.cls[0].cpu().numpy())
                # 获取类别名称
                class_name = class_names[cls]
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'class_id': cls,
                    'class_name': class_name
                })
    
    # 可视化结果
    if detections:
        # 读取图像
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 在图像上绘制检测结果
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']
            
            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # 绘制标签
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # 保存结果图像
        output_path = f"result_{os.path.basename(image_path)}"
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, img)
        print(f"结果已保存至: {output_path}")
    
    return detections

def predict_on_directory(directory_path, model_path='banana_detection_yolov8_final.pt'):
    """
    对目录中的所有图像进行预测
    
    参数:
        directory_path: 图像目录路径
        model_path: 训练好的模型路径
    """
    # 检查目录是否存在
    if not os.path.exists(directory_path):
        print(f"错误: 目录不存在: {directory_path}")
        return
    
    # 支持的图像格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # 获取目录中的所有图像文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(directory_path).glob(f'*{ext}'))
        image_files.extend(Path(directory_path).glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"目录中没有找到图像文件: {directory_path}")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 对每个图像进行预测
    for img_file in image_files:
        print(f"\n处理图像: {img_file}")
        detections = predict_banana_ripeness(str(img_file), model_path)
        
        if detections:
            print(f"检测到 {len(detections)} 个香蕉:")
            for i, det in enumerate(detections):
                print(f"  {i+1}. {det['class_name']} (置信度: {det['confidence']:.2f})")
        else:
            print("未检测到香蕉")

# 示例使用
if __name__ == "__main__":
    # 检查是否有训练好的模型
    model_path = 'banana_detection_yolov8_final.pt'
    if not os.path.exists(model_path):
        print(f"警告: 模型文件不存在: {model_path}")
        print("请先运行训练脚本 train_yolov8.py")
        exit(1)
    
    # 示例1: 预测单张图像
    image_path = 'test_image.jpg'  # 替换为实际的图像路径
    if os.path.exists(image_path):
        print(f"预测图像: {image_path}")
        detections = predict_banana_ripeness(image_path, model_path)
        
        if detections:
            print(f"检测到 {len(detections)} 个香蕉:")
            for i, det in enumerate(detections):
                print(f"  {i+1}. {det['class_name']} (置信度: {det['confidence']:.2f})")
        else:
            print("未检测到香蕉")
    else:
        print(f"测试图像不存在: {image_path}")
    
    # 示例2: 预测目录中的所有图像
    test_dir = 'archive/Banana Ripeness Classification Dataset/test'  # 替换为实际的目录路径
    if os.path.exists(test_dir):
        print(f"\n预测目录中的图像: {test_dir}")
        predict_on_directory(test_dir, model_path)