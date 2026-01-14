import streamlit as st
import torch
import cv2
import os
from pathlib import Path
from PIL import Image
import numpy as np
import tempfile
import matplotlib.pyplot as plt
from collections import defaultdict

# 配置Streamlit
st.set_page_config(
    page_title="香蕉成熟度检测系统",
    page_icon="🍌",
    layout="wide"
)

# 页面标题
st.title("香蕉成熟度检测系统")
st.write("使用YOLOv8模型和颜色特征分析检测香蕉的成熟度")

# 侧边栏
st.sidebar.title("模型设置")
model_path = st.sidebar.text_input("模型路径", value="banana_detection_yolov8_final.pt")
# 降低默认置信度阈值，从0.5改为0.3，提高检测灵敏度
confidence_threshold = st.sidebar.slider("置信度阈值", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
use_color_analysis = st.sidebar.checkbox("启用颜色特征分析", value=True)

# 添加图像增强选项
enhance_image = st.sidebar.checkbox("启用图像增强", value=True)
enhance_factor = st.sidebar.slider("图像增强强度", min_value=1.0, max_value=2.0, value=1.2, step=0.1)

# 类别标签
class_names = ['overripe', 'ripe', 'rotten', 'unripe']
class_descriptions = {
    'overripe': '过熟',
    'ripe': '成熟',
    'rotten': '腐烂',
    'unripe': '未熟'
}


def enhance_image_for_detection(image):
    """增强图像以提高检测效果"""
    if not enhance_image:
        return image

    # 转换为LAB颜色空间进行亮度调整
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 应用CLAHE (限制对比度自适应直方图均衡化)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # 合并通道并转回BGR
    enhanced_lab = cv2.merge((l, a, b))
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # 轻微调整对比度和亮度
    enhanced_image = cv2.convertScaleAbs(enhanced_image, alpha=enhance_factor, beta=10)

    return enhanced_image


# 颜色特征提取函数
def extract_color_features(image):
    """提取图像的颜色特征，用于区分香蕉成熟度"""
    # 转换为HSV颜色空间，更适合颜色分析
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 计算图像的平均色调、饱和度和亮度
    mean_hsv = np.mean(hsv, axis=(0, 1))

    # 计算黄色区域的像素比例
    # 香蕉的黄色范围在HSV中大约是(20-30, 100-255, 100-255)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_ratio = np.sum(yellow_mask > 0) / (image.shape[0] * image.shape[1])

    # 计算绿色区域的像素比例（未成熟香蕉）
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.sum(green_mask > 0) / (image.shape[0] * image.shape[1])

    # 计算棕色/黑色区域的像素比例（过熟/腐烂香蕉）
    lower_brown = np.array([8, 60, 20])
    upper_brown = np.array([20, 255, 200])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    brown_ratio = np.sum(brown_mask > 0) / (image.shape[0] * image.shape[1])

    # 计算图像的亮度分布
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness_mean = np.mean(gray)
    brightness_std = np.std(gray)

    # 返回颜色特征
    return {
        'mean_hue': mean_hsv[0],
        'mean_saturation': mean_hsv[1],
        'mean_value': mean_hsv[2],
        'yellow_ratio': yellow_ratio,
        'green_ratio': green_ratio,
        'brown_ratio': brown_ratio,
        'brightness_mean': brightness_mean,
        'brightness_std': brightness_std
    }


def classify_by_color_features(image_path):
    """基于颜色特征对香蕉进行分类"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        return "no_banana", 0.0

    # 应用图像增强
    image = enhance_image_for_detection(image)

    # 提取颜色特征
    features = extract_color_features(image)

    # 调整阈值，使其更适合摄像头拍摄的图像
    # 降低绿色区域阈值，提高未成熟香蕉的检测率
    if features['green_ratio'] > 0.10 and features['yellow_ratio'] < 0.4:
        # 如果绿色区域多，黄色区域少，可能是未成熟
        confidence = min(0.9, features['green_ratio'] * 3)
        return "unripe", confidence
    # 降低棕色区域阈值，提高过熟/腐烂香蕉的检测率
    elif features['brown_ratio'] > 0.15:
        # 如果棕色区域多，可能是过熟或腐烂
        if features['brightness_mean'] < 100:
            # 如果图像较暗，可能是腐烂
            confidence = min(0.9, features['brown_ratio'] * 3)
            return "rotten", confidence
        else:
            # 如果图像较亮，可能是过熟
            confidence = min(0.9, features['brown_ratio'] * 2.5)
            return "overripe", confidence
    # 降低黄色区域阈值，提高成熟香蕉的检测率
    elif features['yellow_ratio'] > 0.3:
        # 如果黄色区域多，可能是成熟
        confidence = min(0.9, features['yellow_ratio'] * 2)
        return "ripe", confidence
    else:
        # 其他情况，但不直接返回"no_banana"，而是尝试更宽松的判断
        # 如果有任何显著的颜色特征，尝试分类
        max_ratio = max(features['yellow_ratio'], features['green_ratio'], features['brown_ratio'])
        if max_ratio > 0.08:  # 降低阈值
            if max_ratio == features['green_ratio']:
                return "unripe", max_ratio * 2
            elif max_ratio == features['brown_ratio']:
                return "overripe" if features['brightness_mean'] > 100 else "rotten", max_ratio * 2
            else:  # yellow_ratio
                return "ripe", max_ratio * 2
        return "no_banana", 0.0


def create_color_analysis_chart(image, features):
    """创建颜色特征分析图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 显示原始图像
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax1.imshow(image_rgb)
    ax1.set_title("原始图像")
    ax1.axis('off')

    # 显示颜色特征条形图
    feature_names = ['黄色', '绿色', '棕色']
    feature_values = [features['yellow_ratio'], features['green_ratio'], features['brown_ratio']]
    colors = ['gold', 'green', 'brown']

    ax2.bar(feature_names, feature_values, color=colors)
    ax2.set_title('颜色比例分析')
    ax2.set_ylim(0, max(0.5, max(feature_values) * 1.2))
    ax2.set_ylabel('比例')

    plt.tight_layout()
    return fig


# 检查模型是否存在
if not os.path.exists(model_path):
    st.sidebar.error(f"模型文件不存在: {model_path}")
    st.info("请先运行训练脚本 train_yolov8.py 生成模型文件")
    st.stop()


# 加载模型
@st.cache_resource
def load_model(model_path):
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"加载模型失败: {str(e)}")
        return None


model = load_model(model_path)
if model is None:
    st.stop()

# 主界面
st.write("上传图像或使用摄像头进行检测")

# 选项卡
tab1, tab2 = st.tabs(["图像上传", "摄像头"])

with tab1:
    # 图像上传
    uploaded_file = st.file_uploader("选择图像文件", type=['jpg', 'jpeg', 'png', 'bmp'])

    if uploaded_file is not None:
        # 读取上传的图像
        image = Image.open(uploaded_file)
        st.image(image, caption="上传的图像", use_column_width=True)

        # 进行预测
        if st.button("检测香蕉成熟度"):
            with st.spinner("正在检测..."):
                # 保存临时图像
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    image.save(tmp.name)
                    tmp_path = tmp.name

                # 应用图像增强
                if enhance_image:
                    img = cv2.imread(tmp_path)
                    enhanced_img = enhance_image_for_detection(img)
                    cv2.imwrite(tmp_path, enhanced_img)

                # 进行预测
                results = model.predict(tmp_path, conf=confidence_threshold)

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

                # 如果启用了颜色分析
                color_analysis_results = None
                if use_color_analysis:
                    color_class, color_confidence = classify_by_color_features(tmp_path)
                    if color_class != "no_banana":
                        color_analysis_results = {
                            'class_name': color_class,
                            'confidence': color_confidence
                        }

                # 确定最终分类
                final_detections = []
                if detections:
                    for det in detections:
                        final_class = det['class_name']
                        final_confidence = det['confidence']

                        # 如果启用了颜色分析，并且颜色分析结果置信度高
                        if color_analysis_results and color_analysis_results['confidence'] > 0.6:
                            # 如果YOLO检测到的是成熟或腐烂，但颜色分析表明是未成熟或过熟，优先考虑颜色分析
                            if (det['class_name'] == "ripe" and color_analysis_results['class_name'] in ["unripe",
                                                                                                         "overripe"]) or \
                                    (det['class_name'] == "rotten" and color_analysis_results[
                                        'class_name'] == "overripe"):
                                final_class = color_analysis_results['class_name']
                                final_confidence = color_analysis_results['confidence']

                        final_detections.append({
                            'bbox': det['bbox'],
                            'confidence': final_confidence,
                            'class_id': det['class_id'],
                            'class_name': final_class,
                            'original_class': det['class_name'],
                            'color_override': color_analysis_results and final_class != det['class_name']
                        })
                else:
                    # 如果YOLO没有检测到香蕉，但颜色分析检测到香蕉
                    # 降低置信度阈值，从0.5改为0.3，提高回退检测的成功率
                    if color_analysis_results and color_analysis_results['confidence'] > 0.3:
                        final_detections.append({
                            'bbox': None,  # 没有边界框
                            'confidence': color_analysis_results['confidence'],
                            'class_id': None,
                            'class_name': color_analysis_results['class_name'],
                            'original_class': None,
                            'color_override': True
                        })

                # 可视化结果
                if final_detections:
                    # 读取图像
                    img = cv2.imread(tmp_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # 在图像上绘制检测结果
                    for det in final_detections:
                        if det['bbox']:
                            x1, y1, x2, y2 = det['bbox']
                            conf = det['confidence']
                            class_name = det['class_name']

                            # 根据是否被颜色分析覆盖选择不同颜色
                            color = (0, 255, 0) if det['color_override'] else (255, 0, 0)

                            # 绘制边界框
                            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                            # 绘制标签
                            label = f"{class_descriptions[class_name]}: {conf:.2f}"
                            if det['color_override']:
                                label += " (颜色分析)"
                            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # 显示结果图像
                    st.image(img, caption="检测结果", use_column_width=True)

                    # 显示检测结果
                    st.write("检测结果:")
                    for i, det in enumerate(final_detections):
                        if det['bbox']:
                            st.write(
                                f"{i + 1}. {class_descriptions[det['class_name']]} (置信度: {det['confidence']:.2f})")
                            if det['color_override']:
                                st.write(
                                    f"   - 原始YOLO结果: {class_descriptions[det['original_class']]} (已被颜色分析覆盖)")
                        else:
                            st.write(
                                f"{i + 1}. {class_descriptions[det['class_name']]} (置信度: {det['confidence']:.2f}) (仅颜色分析)")

                    # 如果启用了颜色分析，显示颜色特征分析图
                    if use_color_analysis and color_analysis_results:
                        image_for_analysis = cv2.imread(tmp_path)
                        features = extract_color_features(image_for_analysis)
                        fig = create_color_analysis_chart(image_for_analysis, features)
                        st.pyplot(fig)

                        # 显示颜色特征详细信息
                        st.write("颜色特征分析:")
                        st.write(f"- 黄色区域比例: {features['yellow_ratio']:.2%}")
                        st.write(f"- 绿色区域比例: {features['green_ratio']:.2%}")
                        st.write(f"- 棕色区域比例: {features['brown_ratio']:.2%}")
                        st.write(f"- 平均亮度: {features['brightness_mean']:.2f}")
                else:
                    st.write("未检测到香蕉")

                # 删除临时文件
                os.unlink(tmp_path)

with tab2:
    # 摄像头
    st.write("使用摄像头进行实时检测")
    camera_image = st.camera_input("拍照")

    if camera_image is not None:
        # 读取摄像头图像
        image = Image.open(camera_image)
        st.image(image, caption="摄像头图像", use_column_width=True)

        # 进行预测
        if st.button("检测香蕉成熟度", key="camera"):
            with st.spinner("正在检测..."):
                # 保存临时图像
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    image.save(tmp.name)
                    tmp_path = tmp.name

                # 应用图像增强
                if enhance_image:
                    img = cv2.imread(tmp_path)
                    enhanced_img = enhance_image_for_detection(img)
                    cv2.imwrite(tmp_path, enhanced_img)

                # 进行预测
                results = model.predict(tmp_path, conf=confidence_threshold)

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

                # 如果启用了颜色分析
                color_analysis_results = None
                if use_color_analysis:
                    color_class, color_confidence = classify_by_color_features(tmp_path)
                    if color_class != "no_banana":
                        color_analysis_results = {
                            'class_name': color_class,
                            'confidence': color_confidence
                        }

                # 确定最终分类
                final_detections = []
                if detections:
                    for det in detections:
                        final_class = det['class_name']
                        final_confidence = det['confidence']

                        # 如果启用了颜色分析，并且颜色分析结果置信度高
                        if color_analysis_results and color_analysis_results['confidence'] > 0.6:
                            # 如果YOLO检测到的是成熟或腐烂，但颜色分析表明是未成熟或过熟，优先考虑颜色分析
                            if (det['class_name'] == "ripe" and color_analysis_results['class_name'] in ["unripe",
                                                                                                         "overripe"]) or \
                                    (det['class_name'] == "rotten" and color_analysis_results[
                                        'class_name'] == "overripe"):
                                final_class = color_analysis_results['class_name']
                                final_confidence = color_analysis_results['confidence']

                        final_detections.append({
                            'bbox': det['bbox'],
                            'confidence': final_confidence,
                            'class_id': det['class_id'],
                            'class_name': final_class,
                            'original_class': det['class_name'],
                            'color_override': color_analysis_results and final_class != det['class_name']
                        })
                else:
                    # 如果YOLO没有检测到香蕉，但颜色分析检测到香蕉
                    # 降低置信度阈值，从0.5改为0.3，提高回退检测的成功率
                    if color_analysis_results and color_analysis_results['confidence'] > 0.3:
                        final_detections.append({
                            'bbox': None,  # 没有边界框
                            'confidence': color_analysis_results['confidence'],
                            'class_id': None,
                            'class_name': color_analysis_results['class_name'],
                            'original_class': None,
                            'color_override': True
                        })

                # 可视化结果
                if final_detections:
                    # 读取图像
                    img = cv2.imread(tmp_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # 在图像上绘制检测结果
                    for det in final_detections:
                        if det['bbox']:
                            x1, y1, x2, y2 = det['bbox']
                            conf = det['confidence']
                            class_name = det['class_name']

                            # 根据是否被颜色分析覆盖选择不同颜色
                            color = (0, 255, 0) if det['color_override'] else (255, 0, 0)

                            # 绘制边界框
                            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                            # 绘制标签
                            label = f"{class_descriptions[class_name]}: {conf:.2f}"
                            if det['color_override']:
                                label += " (颜色分析)"
                            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # 显示结果图像
                    st.image(img, caption="检测结果", use_column_width=True)

                    # 显示检测结果
                    st.write("检测结果:")
                    for i, det in enumerate(final_detections):
                        if det['bbox']:
                            st.write(
                                f"{i + 1}. {class_descriptions[det['class_name']]} (置信度: {det['confidence']:.2f})")
                            if det['color_override']:
                                st.write(
                                    f"   - 原始YOLO结果: {class_descriptions[det['original_class']]} (已被颜色分析覆盖)")
                        else:
                            st.write(
                                f"{i + 1}. {class_descriptions[det['class_name']]} (置信度: {det['confidence']:.2f}) (仅颜色分析)")

                    # 如果启用了颜色分析，显示颜色特征分析图
                    if use_color_analysis and color_analysis_results:
                        image_for_analysis = cv2.imread(tmp_path)
                        features = extract_color_features(image_for_analysis)
                        fig = create_color_analysis_chart(image_for_analysis, features)
                        st.pyplot(fig)

                        # 显示颜色特征详细信息
                        st.write("颜色特征分析:")
                        st.write(f"- 黄色区域比例: {features['yellow_ratio']:.2%}")
                        st.write(f"- 绿色区域比例: {features['green_ratio']:.2%}")
                        st.write(f"- 棕色区域比例: {features['brown_ratio']:.2%}")
                        st.write(f"- 平均亮度: {features['brightness_mean']:.2f}")
                else:
                    st.write("未检测到香蕉")

                # 删除临时文件
                os.unlink(tmp_path)

# 页脚
st.markdown("---")
st.write("使用YOLOv8模型和颜色特征分析训练的香蕉成熟度检测系统")