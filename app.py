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
import time

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
confidence_threshold = st.sidebar.slider("置信度阈值", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
use_color_analysis = st.sidebar.checkbox("启用颜色特征分析", value=True)

# 调试选项
st.sidebar.title("调试选项")
show_debug_info = st.sidebar.checkbox("显示调试信息", value=False)
enhance_contrast = st.sidebar.checkbox("增强对比度", value=True)
low_confidence_mode = st.sidebar.checkbox("低置信度模式", value=False,
                                          help="降低检测阈值，提高检测率但可能增加误检")

st.sidebar.markdown("---")
st.sidebar.markdown("### 关于颜色分析")
st.sidebar.info("""
颜色分析功能通过分析图像中黄色、绿色和棕色区域的相对比例来辅助判断香蕉的成熟度：

- **未成熟 (unripe)**: 绿色区域比例 > 0.4
- **成熟 (ripe)**: 黄色区域比例 > 0.5 且总香蕉颜色比例 > 0.3
- **过熟 (overripe)**: 棕色区域比例 > 0.2 且黄色区域比例 > 0.3
- **腐烂 (rotten)**: 棕色区域比例 > 0.4

当颜色分析置信度 > 0.6 时，会优先考虑颜色分析结果。
""")

st.sidebar.markdown("### 关于形状验证")
st.sidebar.info("""
形状验证功能通过轮廓分析来判断图像中是否包含香蕉形状：

- 长宽比 > 1.5
- 圆形度 < 0.7
- 凸性 > 0.7

只有同时通过颜色分析和形状验证的图像才会被识别为香蕉。
""")

# 类别标签
class_names = ['overripe', 'ripe', 'rotten', 'unripe']
class_descriptions = {
    'overripe': '过熟',
    'ripe': '成熟',
    'rotten': '腐烂',
    'unripe': '未熟'
}


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


def contains_banana_shape(image, min_area=1000):
    """检查图像是否包含香蕉形状的轮廓"""
    # 预处理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 自适应阈值
    binary = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # 形态学操作
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    # 寻找轮廓
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    banana_like_contours = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # 计算轮廓特征
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        # 圆形度
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # 长宽比
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0

        # 凸性
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0

        # 香蕉形状特征：
        # 1. 长宽比 > 1.5（长条形）
        # 2. 圆形度 < 0.7（不是圆形）
        # 3. 凸性 > 0.7（相对平滑）
        if (aspect_ratio > 1.5 and
                circularity < 0.7 and
                convexity > 0.7 and
                area > min_area):
            banana_like_contours += 1

    return banana_like_contours > 0


def classify_by_color_features(image_array):
    """基于颜色特征对香蕉进行分类"""
    if image_array is None:
        return "no_banana", 0.0, False

    # 检查是否有香蕉形状
    has_shape = contains_banana_shape(image_array)

    # 提取颜色特征
    features = extract_color_features(image_array)

    # 计算总香蕉相关颜色比例（黄+绿+棕）
    total_banana_colors = features['yellow_ratio'] + features['green_ratio'] + features['brown_ratio']

    # 安全阈值：如果没有形状且颜色比例很低，直接返回无香蕉
    if not has_shape and total_banana_colors < 0.3:
        return "no_banana", 0.0, has_shape

    # 调整置信度：有形状的置信度更高
    shape_boost = 1.3 if has_shape else 1.0

    # 基于特征进行分类
    # 这些阈值是根据经验设定的，可能需要调整
    if features['green_ratio'] > 0.15 and features['yellow_ratio'] < 0.3:
        # 如果绿色区域多，黄色区域少，可能是未成熟
        confidence = min(0.9, features['green_ratio'] * 3 * shape_boost)
        return "unripe", confidence, has_shape
    elif features['brown_ratio'] > 0.2:
        # 如果棕色区域多，可能是过熟或腐烂
        if features['brightness_mean'] < 100:
            # 如果图像较暗，可能是腐烂
            confidence = min(0.9, features['brown_ratio'] * 3 * shape_boost)
            return "rotten", confidence, has_shape
        else:
            # 如果图像较亮，可能是过熟
            confidence = min(0.9, features['brown_ratio'] * 2.5 * shape_boost)
            return "overripe", confidence, has_shape
    elif features['yellow_ratio'] > 0.5:
        if total_banana_colors > 0.6:
            # 提高黄色阈值要求，并确保总香蕉颜色比例足够高
            # 如果黄色区域多，可能是成熟
            confidence = min(0.9, features['yellow_ratio'] * 2 * shape_boost)
            return "ripe", confidence, has_shape
        else:
            return "no_banana", 0.0, has_shape
    else:
        # 其他情况
        return "no_banana", 0.0, has_shape


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
def process_detections(results, image_array, use_color_analysis=True, low_confidence_mode=False):
    """处理检测结果"""
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
        color_class, color_confidence, has_shape = classify_by_color_features(image_array)
        if color_class != "no_banana":
            color_analysis_results = {
                'class_name': color_class,
                'confidence': color_confidence,
                'has_shape': has_shape
            }

    # 确定最终分类
    final_detections = []
    if detections:
        for det in detections:
            final_class = det['class_name']
            final_confidence = det['confidence']

            # 如果启用了颜色分析，并且颜色分析结果置信度高且有形状验证
            if (color_analysis_results and 
                color_analysis_results['confidence'] > 0.6 and 
                color_analysis_results['has_shape']):
                # 如果YOLO检测到的是成熟或腐烂，但颜色分析表明是未成熟或过熟，优先考虑颜色分析
                if (det['class_name'] == "ripe" and color_analysis_results['class_name'] in ["unripe", "overripe"]) or \
                        (det['class_name'] == "rotten" and color_analysis_results['class_name'] == "overripe"):
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
        # 提高颜色分析置信度阈值，避免误判
        if (color_analysis_results and 
            color_analysis_results['confidence'] > 0.7 and  # 从0.5提高到0.7
            color_analysis_results['has_shape']):  # 添加形状验证要求
            final_detections.append({
                'bbox': None,  # 没有边界框
                'confidence': color_analysis_results['confidence'],
                'class_id': None,
                'class_name': color_analysis_results['class_name'],
                'original_class': None,
                'color_override': True
            })
        # 低置信度模式：即使颜色分析置信度较低也尝试使用
        elif (low_confidence_mode and 
              color_analysis_results and 
              color_analysis_results['confidence'] > 0.5 and  # 从0.3提高到0.5
              color_analysis_results['has_shape']):  # 添加形状验证要求
            final_detections.append({
                'bbox': None,
                'confidence': color_analysis_results['confidence'],
                'class_id': None,
                'class_name': color_analysis_results['class_name'],
                'original_class': None,
                'color_override': True
            })

    return final_detections, color_analysis_results


def visualize_results(image_array, final_detections, class_descriptions):
    """可视化检测结果"""
    img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

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

    return img


def enhance_image_contrast(image):
    """增强图像对比度"""
    if len(image.shape) == 3:
        # 彩色图像
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    else:
        # 灰度图像
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)

    return enhanced


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
    uploaded_file = st.file_uploader("选择图像文件", type=['jpg', 'jpeg', 'png', 'bmp'], key="image_uploader")

    if uploaded_file is not None:
        # 读取上传的图像
        image = Image.open(uploaded_file)
        st.image(image, caption="上传的图像", use_column_width=True)

        # 进行预测
        if st.button("检测香蕉成熟度", key="image_detect"):
            with st.spinner("正在检测..."):
                # 保存临时图像
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    image.save(tmp.name)
                    tmp_path = tmp.name

                # 进行预测
                conf_threshold = 0.3 if low_confidence_mode else confidence_threshold
                results = model.predict(tmp_path, conf=conf_threshold)

                # 读取图像用于处理
                image_array = cv2.imread(tmp_path)

                # 如果需要增强对比度
                if enhance_contrast:
                    image_array = enhance_image_contrast(image_array)

                # 处理检测结果
                final_detections, color_analysis_results = process_detections(
                    results, image_array, use_color_analysis, low_confidence_mode
                )

                # 可视化结果
                if final_detections:
                    # 显示结果图像
                    result_img = visualize_results(image_array, final_detections, class_descriptions)
                    st.image(result_img, caption="检测结果", use_column_width=True)

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
                        features = extract_color_features(image_array)
                        fig = create_color_analysis_chart(image_array, features)
                        st.pyplot(fig)

                        # 显示颜色特征详细信息
                        st.write("颜色特征分析:")
                        st.write(f"- 黄色区域比例: {features['yellow_ratio']:.2%}")
                        st.write(f"- 绿色区域比例: {features['green_ratio']:.2%}")
                        st.write(f"- 棕色区域比例: {features['brown_ratio']:.2%}")
                        st.write(f"- 平均亮度: {features['brightness_mean']:.2f}")
                else:
                    st.write("未检测到香蕉")

                    # 显示调试信息
                    if show_debug_info:
                        st.write("调试信息:")
                        features = extract_color_features(image_array)
                        st.write(f"图像尺寸: {image_array.shape[:2]}")
                        st.write(
                            f"颜色特征: 黄{features['yellow_ratio']:.2%}, 绿{features['green_ratio']:.2%}, 棕{features['brown_ratio']:.2%}")

                # 删除临时文件
                os.unlink(tmp_path)

with tab2:
    # 摄像头
    st.write("使用摄像头进行实时检测")

    # 摄像头使用提示
    st.info("💡 使用提示：")
    st.info("1. 确保香蕉在图像中清晰可见")
    st.info("2. 让香蕉占据画面的主要部分")
    st.info("3. 确保光线充足")
    st.info("4. 尝试不同角度拍摄")

    camera_image = st.camera_input("拍照", key="camera_input")

    if camera_image is not None:
        # 读取摄像头图像
        image = Image.open(camera_image)
        st.image(image, caption="摄像头图像", use_column_width=True)

        # 进行预测
        if st.button("检测香蕉成熟度", key="camera_detect"):
            with st.spinner("正在检测..."):
                # 将PIL图像转换为numpy数组
                image_np = np.array(image)

                # 确保图像是BGR格式（OpenCV格式）
                if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                    # PIL是RGB，OpenCV需要BGR
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                # 保存原始图像尺寸
                original_height, original_width = image_np.shape[:2]

                # 图像预处理
                processed_image = image_np.copy()

                # 1. 增强对比度
                if enhance_contrast:
                    processed_image = enhance_image_contrast(processed_image)

                # 2. 调整图像大小（如果太大）
                max_dimension = 1280
                if max(original_height, original_width) > max_dimension:
                    scale = max_dimension / max(original_height, original_width)
                    new_width = int(original_width * scale)
                    new_height = int(original_height * scale)
                    processed_image = cv2.resize(processed_image, (new_width, new_height))

                # 保存临时图像用于调试
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    cv2.imwrite(tmp.name, processed_image)
                    tmp_path = tmp.name

                # 使用模型进行预测
                conf_threshold = 0.3 if low_confidence_mode else confidence_threshold

                # 第一次检测
                results = model.predict(processed_image, conf=conf_threshold, verbose=False)

                # 处理检测结果
                final_detections, color_analysis_results = process_detections(
                    results, processed_image, use_color_analysis, low_confidence_mode
                )

                # 如果第一次没有检测到，尝试第二次检测（使用不同参数）
                if not final_detections:
                    st.warning("第一次检测未发现香蕉，尝试第二次检测...")

                    # 尝试使用更低的置信度阈值
                    results2 = model.predict(processed_image, conf=0.2, verbose=False)
                    final_detections, color_analysis_results = process_detections(
                        results2, processed_image, use_color_analysis, low_confidence_mode
                    )

                # 可视化结果
                if final_detections:
                    # 如果需要将边界框坐标映射回原始图像尺寸
                    if processed_image.shape != image_np.shape:
                        # 计算缩放比例
                        scale_h = original_height / processed_image.shape[0]
                        scale_w = original_width / processed_image.shape[1]

                        # 调整边界框坐标
                        for det in final_detections:
                            if det['bbox']:
                                x1, y1, x2, y2 = det['bbox']
                                det['bbox'] = [
                                    int(x1 * scale_w),
                                    int(y1 * scale_h),
                                    int(x2 * scale_w),
                                    int(y2 * scale_h)
                                ]

                    # 显示结果图像
                    result_img = visualize_results(image_np, final_detections, class_descriptions)
                    st.image(result_img, caption="检测结果", use_column_width=True)

                    # 显示检测结果
                    st.success("✅ 检测成功！")
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
                        features = extract_color_features(processed_image)
                        fig = create_color_analysis_chart(processed_image, features)
                        st.pyplot(fig)

                        # 显示颜色特征详细信息
                        st.write("颜色特征分析:")
                        st.write(f"- 黄色区域比例: {features['yellow_ratio']:.2%}")
                        st.write(f"- 绿色区域比例: {features['green_ratio']:.2%}")
                        st.write(f"- 棕色区域比例: {features['brown_ratio']:.2%}")
                        st.write(f"- 平均亮度: {features['brightness_mean']:.2f}")
                        st.write(f"- 平均饱和度: {features['mean_saturation']:.2f}")
                else:
                    st.error("❌ 未检测到香蕉")

                    # 显示详细的调试信息
                    if show_debug_info:
                        st.write("### 调试信息")

                        # 显示图像基本信息
                        st.write(f"**图像尺寸:** {original_height} x {original_width}")
                        st.write(f"**处理尺寸:** {processed_image.shape[:2]}")

                        # 显示颜色特征
                        features = extract_color_features(processed_image)
                        st.write("**颜色特征分析:**")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("黄色比例", f"{features['yellow_ratio']:.2%}")
                        with col2:
                            st.metric("绿色比例", f"{features['green_ratio']:.2%}")
                        with col3:
                            st.metric("棕色比例", f"{features['brown_ratio']:.2%}")

                        st.write(f"**平均亮度:** {features['brightness_mean']:.1f}")
                        st.write(f"**平均饱和度:** {features['mean_saturation']:.1f}")

                        # 显示特征图
                        fig, axes = plt.subplots(2, 3, figsize=(12, 8))

                        # 原始图像
                        axes[0, 0].imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
                        axes[0, 0].set_title("原始图像")
                        axes[0, 0].axis('off')

                        # HSV空间
                        hsv = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)
                        axes[0, 1].imshow(hsv[:, :, 0], cmap='hsv')
                        axes[0, 1].set_title("Hue通道")
                        axes[0, 1].axis('off')

                        axes[0, 2].imshow(hsv[:, :, 1], cmap='gray')
                        axes[0, 2].set_title("Saturation通道")
                        axes[0, 2].axis('off')

                        # 颜色掩码
                        # 黄色掩码
                        lower_yellow = np.array([20, 100, 100])
                        upper_yellow = np.array([30, 255, 255])
                        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
                        axes[1, 0].imshow(yellow_mask, cmap='gray')
                        axes[1, 0].set_title(f"黄色区域: {features['yellow_ratio']:.2%}")
                        axes[1, 0].axis('off')

                        # 绿色掩码
                        lower_green = np.array([35, 40, 40])
                        upper_green = np.array([85, 255, 255])
                        green_mask = cv2.inRange(hsv, lower_green, upper_green)
                        axes[1, 1].imshow(green_mask, cmap='gray')
                        axes[1, 1].set_title(f"绿色区域: {features['green_ratio']:.2%}")
                        axes[1, 1].axis('off')

                        # 棕色掩码
                        lower_brown = np.array([8, 60, 20])
                        upper_brown = np.array([20, 255, 200])
                        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
                        axes[1, 2].imshow(brown_mask, cmap='gray')
                        axes[1, 2].set_title(f"棕色区域: {features['brown_ratio']:.2%}")
                        axes[1, 2].axis('off')

                        plt.tight_layout()
                        st.pyplot(fig)

                        # 给出改进建议
                        st.write("### 改进建议")
                        if features['yellow_ratio'] < 0.1 and features['green_ratio'] < 0.1:
                            st.warning("图像中黄色和绿色区域很少，可能不是香蕉或颜色失真")
                            st.info("建议：拍摄更清晰的香蕉图像，确保香蕉占据画面主要部分")
                        elif features['brightness_mean'] < 50:
                            st.warning("图像太暗，可能影响检测")
                            st.info("建议：增加光线或使用闪光灯")
                        elif features['brightness_mean'] > 200:
                            st.warning("图像过曝，可能影响检测")
                            st.info("建议：减少光线或调整角度")
                        else:
                            st.info("图像质量尚可，但模型未能检测到香蕉。可以尝试：")
                            st.info("1. 开启'低置信度模式'")
                            st.info("2. 调整香蕉在画面中的位置")
                            st.info("3. 使用更清晰的香蕉图像")

                # 删除临时文件
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

# 页脚
st.markdown("---")
st.write("使用YOLOv8模型和颜色特征分析训练的香蕉成熟度检测系统")
st.write("**使用说明：**")
st.write("1. 在'图像上传'选项卡中上传香蕉图片进行检测")
st.write("2. 在'摄像头'选项卡中使用摄像头拍摄香蕉进行实时检测")
st.write("3. 如果检测不到香蕉，可以尝试调整侧边栏的调试选项")