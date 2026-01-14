# 香蕉成熟度检测系统

## 项目概述

本项目是一个基于YOLOv8和颜色特征分析的香蕉成熟度检测系统，能够准确识别香蕉的四种成熟度状态：过熟(overripe)、成熟(ripe)、腐烂(rotten)和未熟(unripe)。系统结合了深度学习目标检测和计算机视觉颜色分析技术，提供了高精度的香蕉成熟度检测能力，并包含友好的Web界面供用户使用。

## 项目特点

- **双重检测机制**：结合YOLOv8目标检测和颜色特征分析，提高检测准确性
- **Web应用界面**：基于Streamlit的直观用户界面，支持图像上传和实时摄像头检测
- **批量处理**：支持批量图像分类和结果可视化
- **详细分析**：提供颜色特征分析图表，帮助理解检测结果
- **多种使用方式**：支持命令行、Python脚本和Web应用三种使用方式

## 项目结构

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
│   ├── 📁 "Banana Ripeness Classification Dataset"/
│   │   ├── 📁 train/                  # 训练集
│   │   │   ├── 📁 overripe/           # 过熟香蕉图像
│   │   │   ├── 📁 ripe/               # 成熟香蕉图像
│   │   │   ├── 📁 rotten/             # 腐烂香蕉图像
│   │   │   └── 📁 unripe/             # 未熟香蕉图像
│   │   ├── 📁 test/                   # 测试集
│   │   └── 📁 val/                    # 验证集
│   ├── app.py                         # 原始Web应用
│   ├── classify_test_images.py        # 基础分类脚本
│   ├── improved_classification.py     # 改进的分类脚本（结合颜色特征）
│   └── train_yolov8.py                # 训练脚本
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
│   ├── img.png                        # 测试图像
│   ├── img_1.png                      # 测试图像
│   └── ...                            # 其他测试图像
│
├── 📁 classified_results/             # 基础分类结果
│   ├── 📁 overripe/                   # 过熟香蕉图像
│   ├── 📁 ripe/                       # 成熟香蕉图像
│   ├── 📁 rotten/                     # 腐烂香蕉图像
│   ├── 📁 unripe/                     # 未熟香蕉图像
│   ├── 📁 no_banana/                  # 无香蕉图像
│   ├── classification_results.csv     # 分类结果CSV
│   ├── classification_chart.png       # 分类结果饼图
│   └── sample_collage.jpg             # 样本拼贴图
│
├── 📁 improved_classification_results/ # 改进分类结果
│   ├── 📁 overripe/                   # 过熟香蕉图像
│   ├── 📁 ripe/                       # 成熟香蕉图像
│   ├── 📁 rotten/                     # 腐烂香蕉图像
│   ├── 📁 unripe/                     # 未熟香蕉图像
│   ├── 📁 no_banana/                  # 无香蕉图像
│   ├── improved_classification_results.csv # 改进分类结果CSV
│   ├── classification_chart.png       # 分类结果饼图
│   └── color_analysis_chart.png       # 颜色分析图表
│
├── 🐍 Python脚本文件
│   ├── app.py                         # Streamlit Web应用主程序（从archive复制）
│   ├── predict_banana.py              # 香蕉预测脚本
│   ├── test_model.py                  # 模型测试脚本
│   ├── train.py                       # 基础训练脚本
│   ├── train_yolo.py                  # YOLO训练脚本
│   ├── train_yolo_api.py              # YOLO API训练脚本
│   ├── train_yolo_cfg.py              # YOLO配置训练脚本
│   ├── train_yolo_custom.py           # YOLO自定义训练脚本
│   ├── train_yolo_simple.py           # YOLO简化训练脚本
│   ├── create_yolo_dataset.py          # 创建YOLO数据集脚本
│   └── info_yolo.py                    # YOLO信息脚本
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
    ├── README.md                      # 项目主文档（本文件）
    └── PROJECT_STRUCTURE.md           # 项目结构文档
```

## 环境要求

- Python 3.9+
- PyTorch
- Ultralytics (YOLOv8)
- OpenCV
- Streamlit (用于Web应用)
- Matplotlib (用于可视化)
- NumPy
- Pandas

## 安装依赖

```bash
# 创建虚拟环境（如果尚未创建）
python -m venv .venv

# 激活虚拟环境
.venv\Scripts\activate

# 安装依赖
pip install torch torchvision ultralytics opencv-python streamlit matplotlib pandas
```

## 数据集准备

数据集已包含在`archive/Banana Ripeness Classification Dataset`目录中，包含以下结构：

- **训练集**：`train/overripe`, `train/ripe`, `train/rotten`, `train/unripe`
- **测试集**：`test/overripe`, `test/ripe`, `test/rotten`, `test/unripe`
- **验证集**：`val/overripe`, `val/ripe`, `val/rotten`, `val/unripe`

每个目录包含对应类别的图像和YOLO格式的标注文件(.txt)。

## 训练模型

运行以下命令训练模型：

```bash
python achive/train_yolov8.py
```

训练完成后，模型将保存为`banana_detection_yolov8_final.pt`。

## 使用模型进行预测

### 1. 命令行预测

```bash
# 预测单张图像
python predict_banana.py

# 预测目录中的所有图像
python predict_banana.py --directory path/to/images
```

### 2. 批量分类

#### 基础分类（仅使用YOLOv8）
```bash
python achive/classify_test_images.py
```

#### 改进分类（结合颜色特征分析）
```bash
python achive/improved_classification.py
```

### 3. Web应用

运行以下命令启动Web应用：

```bash
streamlit run app.py
```

然后在浏览器中打开显示的URL（通常是`http://localhost:8501`）。

## Web应用使用指南

### 启动应用

1. 确保已激活虚拟环境：`.venv\Scripts\activate`
2. 运行命令：`streamlit run app.py`
3. 在浏览器中打开显示的URL（通常是`http://localhost:8501`）

### 使用界面

1. **侧边栏设置**：
   - 模型路径（默认为"banana_detection_yolov8_final.pt"）
   - 置信度阈值（可调整检测敏感度）
   - 启用颜色特征分析（提高检测准确性）

2. **图像上传选项卡**：
3. 
   - 上传香蕉图片（支持jpg、jpeg、png、bmp格式）
   - 点击"检测香蕉成熟度"按钮进行分析

3. **摄像头选项卡**：
   - 使用摄像头拍摄香蕉图片
   - 点击"检测香蕉成熟度"按钮进行分析

4. **查看结果**：
   - 显示带有边界框的标注图像
   - 显示检测结果列表（成熟度类型和置信度）
   - 如果启用了颜色分析，还会显示颜色特征分析图表

## 类别说明

- **overripe**: 过熟香蕉 - 表皮出现大量褐色斑点，果肉开始软化
- **ripe**: 成熟香蕉 - 表皮呈亮黄色，适合食用
- **rotten**: 腐烂香蕉 - 表皮发黑，果肉变质
- **unripe**: 未熟香蕉 - 表皮呈绿色或黄绿色，果肉较硬

## 颜色特征分析

系统通过分析图像的HSV颜色空间特征来辅助判断香蕉成熟度：

1. **黄色区域比例**：成熟香蕉的主要特征
2. **绿色区域比例**：未熟香蕉的主要特征
3. **棕色区域比例**：过熟或腐烂香蕉的主要特征
4. **亮度分析**：区分过熟和腐烂香蕉的辅助特征

当颜色特征分析的置信度高于0.6时，系统会优先使用颜色分析结果，而不是仅依赖YOLOv8的检测结果，从而提高整体准确性。

## 模型性能

训练完成后，模型在验证集上的性能指标将显示在训练日志中，包括：

- **mAP50**: 平均精度（IoU阈值0.5）
- **mAP50-95**: 平均精度（IoU阈值0.5-0.95）
- **精确率(Precision)**: 预测为正例中实际为正例的比例
- **召回率(Recall)**: 实际正例中被正确预测为正例的比例

## 技术实现细节

### YOLOv8模型

- 使用YOLOv8n（nano版本）作为基础模型
- 输入图像尺寸：320×320像素
- 训练轮数：10（可根据需要调整）
- 批处理大小：8（可根据硬件调整）

### 颜色特征分析

- 颜色空间：HSV（更适合颜色分析）
- 特征提取：
  - 平均色调、饱和度和亮度
  - 黄色、绿色和棕色区域的比例
  - 图像亮度分布

### 双重检测机制

1. 首先使用YOLOv8进行目标检测和初步分类
2. 然后进行颜色特征分析
3. 当颜色特征分析的置信度高于0.6时，优先使用颜色分析结果
4. 否则，使用YOLOv8的检测结果

## 故障排除

如果遇到问题，请检查：

1. **虚拟环境**：是否正确激活虚拟环境
2. **依赖安装**：所有依赖是否正确安装
3. **数据集路径**：数据集路径是否正确配置在`banana_dataset.yaml`中
4. **模型文件**：模型文件是否存在且可访问
5. **硬件资源**：训练可能需要较长时间，确保有足够的计算资源

## 扩展功能

系统支持以下扩展功能：

1. **自定义数据集**：可以通过修改`banana_dataset.yaml`配置文件使用自定义数据集
2. **模型微调**：可以在预训练模型基础上进行微调，提高特定场景下的准确性
3. **实时检测**：Web应用支持实时摄像头检测
4. **批量处理**：支持批量图像处理和结果导出

## 联系方式

如有问题或建议，请提交Issue或Pull Request。