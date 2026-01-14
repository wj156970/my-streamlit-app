# 香蕉成熟度检测系统 - 项目结构文档

## 项目概述
本项目是一个基于YOLOv8的香蕉成熟度检测系统，能够识别香蕉的四种成熟度状态：过熟(overripe)、成熟(ripe)、腐烂(rotten)和未熟(unripe)。

## 完整项目结构

```
Banana_detection/
│
├── 📁 .venv/                          # Python虚拟环境
│   ├── 📁 Lib\site-packages/         # 已安装的依赖包
│   │   ├── ultralytics/               # YOLOv8框架
│   │   ├── streamlit/                 # Web应用框架
│   │   ├── torch/                     # PyTorch深度学习框架
│   │   ├── opencv-python/             # 计算机视觉库
│   │   └── ...                        # 其他依赖包
│   └── 📁 Scripts/                    # Python脚本和可执行文件
│
├── 📁 archive/                        # 数据集归档
│   └── 📁 "Banana Ripeness Classification Dataset"/
│       ├── 📁 train/                  # 训练集
│       │   ├── 📁 overripe/           # 过熟香蕉 (约200张图像)
│       │   ├── 📁 ripe/               # 成熟香蕉 (约200张图像)
│       │   ├── 📁 rotten/             # 腐烂香蕉 (约200张图像)
│       │   └── 📁 unripe/             # 未熟香蕉 (约200张图像)
│       ├── 📁 test/                   # 测试集
│       │   ├── 📁 overripe/
│       │   ├── 📁 ripe/
│       │   ├── 📁 rotten/
│       │   └── 📁 unripe/
│       └── 📁 val/                    # 验证集
│           ├── 📁 overripe/
│           ├── 📁 ripe/
│           ├── 📁 rotten/
│           └── 📁 unripe/
│
├── 📁 runs/                           # 训练结果和日志
│   ├── 📁 detect/                     # 检测结果
│   │   ├── 📁 banana_detection_yolov8/# YOLOv8训练结果
│   │   │   ├── 📁 weights/            # 模型权重文件
│   │   │   │   ├── best.pt            # 最佳模型权重
│   │   │   │   └── last.pt            # 最终模型权重
│   │   │   ├── results.csv            # 训练过程数据
│   │   │   ├── results.png            # 训练结果图表
│   │   │   ├── confusion_matrix.png   # 混淆矩阵
│   │   │   ├── BoxF1_curve.png       # F1曲线
│   │   │   ├── BoxPR_curve.png       # PR曲线
│   │   │   ├── train_batch*.jpg      # 训练批次可视化
│   │   │   └── val_batch*.jpg        # 验证批次可视化
│   │   └── 📁 val/                    # 验证结果
│   └── 📁 train/                      # 训练日志
│       ├── 📁 banana_detection_model*/# 各种训练尝试
│       └── ...
│
├── 📁 test/                           # 测试图像文件夹
│   ├── img.png                        # 测试图像1
│   ├── img_1.png                      # 测试图像2
│   ├── img_2.png                      # 测试图像3
│   ├── img_3.png                      # 测试图像4
│   ├── img_4.png                      # 测试图像5
│   ├── img_5.png                      # 测试图像6
│   └── img_6.png                      # 测试图像7
│
├── 📁 __pycache__/                    # Python缓存文件
│
├── 🐍 Python脚本文件
│   ├── app.py                         # Streamlit Web应用主程序
│   ├── predict_banana.py              # 香蕉预测脚本
│   ├── test_model.py                  # 模型测试脚本
│   ├── train.py                       # 基础训练脚本
│   ├── train_yolo.py                  # YOLO训练脚本
│   ├── train_yolo_api.py              # YOLO API训练脚本
│   ├── train_yolo_cfg.py              # YOLO配置训练脚本
│   ├── train_yolo_custom.py           # YOLO自定义训练脚本
│   ├── train_yolo_simple.py           # YOLO简化训练脚本
│   └── train_yolov8.py                # YOLOv8训练脚本
│
├── 📋 配置文件
│   ├── banana_dataset.yaml            # 数据集配置文件
│   ├── banana_yolov5s.yaml          # YOLOv5模型配置
│   └── dataset.yaml                   # 数据集配置（备用）
│
├── 🎯 预训练模型文件
│   ├── yolov5s.pt                     # YOLOv5预训练模型
│   ├── yolov8n.pt                     # YOLOv8预训练模型
│   └── banana_detection_yolov8_final.pt # 训练完成的香蕉检测模型
│
└── 📄 文档文件
    ├── README.md                      # 项目主文档
    └── PROJECT_STRUCTURE.md           # 项目结构文档（本文件）
```

## 核心文件功能说明

### 训练相关文件
- **train_yolov8.py**: 主要的YOLOv8训练脚本，包含数据集验证、设备配置和训练逻辑
- **banana_dataset.yaml**: 数据集配置文件，定义了训练、验证和测试数据的路径以及类别信息
- **banana_yolov5s.yaml**: YOLOv5模型配置文件（备用）

### 预测相关文件
- **predict_banana.py**: 香蕉成熟度预测脚本，支持单张图像和批量图像预测
- **test_model.py**: 模型测试脚本，用于验证模型加载和基本功能

### Web应用
- **app.py**: Streamlit Web应用，提供用户友好的界面进行香蕉成熟度检测

### 模型文件
- **banana_detection_yolov8_final.pt**: 训练完成的香蕉成熟度检测模型，包含4个类别的识别能力
- **yolov8n.pt**: YOLOv8预训练模型（nano版本）
- **yolov5s.pt**: YOLOv5预训练模型（small版本）

## 类别定义

模型可以识别以下4种香蕉成熟度状态：

1. **overripe** (过熟): 香蕉表皮出现大量褐色斑点，果肉开始软化
2. **ripe** (成熟): 香蕉表皮呈亮黄色，适合食用
3. **rotten** (腐烂): 香蕉表皮发黑，果肉变质
4. **unripe** (未熟): 香蕉表皮呈绿色或黄绿色，果肉较硬

## 训练结果

模型在训练过程中生成了以下关键文件：

- **best.pt**: 在验证集上表现最佳的模型权重
- **last.pt**: 训练结束时最终的模型权重
- **results.csv**: 包含训练过程中的损失值和评估指标
- **confusion_matrix.png**: 混淆矩阵，显示各类别的分类准确率
- **results.png**: 训练曲线图，展示损失值和评估指标的变化趋势

## 使用方法

### 1. 训练模型
```bash
python train_yolov8.py
```

### 2. 测试模型
```bash
python test_model.py
```

### 3. 批量预测
```bash
python predict_banana.py --directory path/to/images
```

### 4. 启动Web应用
```bash
streamlit run app.py
```

## 技术栈

- **深度学习框架**: PyTorch + Ultralytics YOLOv8
- **计算机视觉**: OpenCV
- **Web框架**: Streamlit
- **数据处理**: NumPy, Pandas
- **可视化**: Matplotlib

## 项目特点

1. **多模型支持**: 支持YOLOv5和YOLOv8两种框架
2. **完整的训练流程**: 从数据准备到模型部署的全流程支持
3. **Web界面**: 提供友好的用户界面，支持图像上传和实时检测
4. **批量处理**: 支持批量图像处理和结果导出
5. **详细的可视化**: 提供训练过程的可视化分析和结果展示

## 注意事项

1. 确保虚拟环境已激活：`.venv\Scripts\activate`
2. 所有依赖包已正确安装
3. 数据集路径配置正确
4. 模型文件存在且可访问
5. 训练过程可能需要较长时间，建议使用GPU加速