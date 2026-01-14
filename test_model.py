import torch
import os
from pathlib import Path

# 检查模型文件是否存在
model_path = 'banana_detection_yolov8_final.pt'
if not os.path.exists(model_path):
    print(f"错误: 模型文件不存在: {model_path}")
    exit(1)

# 加载模型
try:
    from ultralytics import YOLO
    model = YOLO(model_path)
    print("成功加载模型")
    
    # 获取模型信息
    print(f"模型类别数量: {len(model.names)}")
    print("类别名称:")
    for i, name in model.names.items():
        print(f"  {i}: {name}")
    
    # 检查测试图像
    test_dir = Path('archive/Banana Ripeness Classification Dataset/test1')
    if test_dir.exists():
        # 找一张测试图像
        for class_dir in test_dir.iterdir():
            if class_dir.is_dir():
                images = list(class_dir.glob('*.jpg'))
                if images:
                    test_image = str(images[0])
                    print(f"\n测试图像: {test_image}")
                    
                    # 进行预测
                    results = model.predict(test_image, conf=0.5)
                    
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
                                class_name = model.names[cls]
                                
                                detections.append({
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'confidence': float(conf),
                                    'class_id': cls,
                                    'class_name': class_name
                                })
                    
                    if detections:
                        print("检测结果:")
                        for i, det in enumerate(detections):
                            print(f"  {i+1}. {det['class_name']} (置信度: {det['confidence']:.2f})")
                    else:
                        print("未检测到香蕉")
                    break
    else:
        print(f"测试目录不存在: {test_dir}")
        
except Exception as e:
    print(f"加载模型失败: {str(e)}")