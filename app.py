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
st.title("🍌 香蕉成熟度检测系统")
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
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_ratio = np.sum(yellow_mask > 0) / (image.shape[0] * image.shape[1])

    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.sum(green_mask > 0) / (image.shape[0] * image.shape[1])

    lower_brown = np.array([8, 60, 20])
    upper_brown = np.array([20, 255, 200])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    brown_ratio = np.sum(brown_mask > 0) / (image.shape[0] * image.shape[1])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness_mean = np.mean(gray)

    return {
        'yellow_ratio': yellow_ratio,
        'green_ratio': green_ratio,
        'brown_ratio': brown_ratio,
        'brightness_mean': brightness_mean,
        'mean_saturation': np.mean(hsv[:, :, 1])
    }


def contains_banana_shape(image, min_area=1000):
    """检查图像是否包含香蕉形状的轮廓"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0

        if (aspect_ratio > 1.5 and
                circularity < 0.7 and
                convexity > 0.7 and
                area > min_area):
            return True
    return False


def classify_by_color_features(image_array):
    """基于颜色特征对香蕉进行分类"""
    if image_array is None:
        return "no_banana", 0.0, False

    has_shape = contains_banana_shape(image_array)
    features = extract_color_features(image_array)
    total_banana_colors = features['yellow_ratio'] + features['green_ratio'] + features['brown_ratio']

    if not has_shape and total_banana_colors < 0.3:
        return "no_banana", 0.0, has_shape

    shape_boost = 1.3 if has_shape else 1.0

    if features['green_ratio'] > 0.15 and features['yellow_ratio'] < 0.3:
        confidence = min(0.9, features['green_ratio'] * 3 * shape_boost)
        return "unripe", confidence, has_shape
    elif features['brown_ratio'] > 0.2:
        if features['brightness_mean'] < 100:
            confidence = min(0.9, features['brown_ratio'] * 3 * shape_boost)
            return "rotten", confidence, has_shape
        else:
            confidence = min(0.9, features['brown_ratio'] * 2.5 * shape_boost)
            return "overripe", confidence, has_shape
    elif features['yellow_ratio'] > 0.5 and total_banana_colors > 0.6:
        confidence = min(0.9, features['yellow_ratio'] * 2 * shape_boost)
        return "ripe", confidence, has_shape
    else:
        return "no_banana", 0.0, has_shape


def create_color_analysis_chart(image, features):
    """创建颜色特征分析图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax1.imshow(image_rgb)
    ax1.set_title("原始图像")
    ax1.axis('off')

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
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                class_name = class_names[cls]

                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'class_id': cls,
                    'class_name': class_name
                })

    color_analysis_results = None
    if use_color_analysis:
        color_class, color_confidence, has_shape = classify_by_color_features(image_array)
        if color_class != "no_banana":
            color_analysis_results = {
                'class_name': color_class,
                'confidence': color_confidence,
                'has_shape': has_shape
            }

    final_detections = []
    if detections:
        for det in detections:
            final_class = det['class_name']
            final_confidence = det['confidence']

            if (color_analysis_results and
                    color_analysis_results['confidence'] > 0.6 and
                    color_analysis_results['has_shape']):
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
        if (color_analysis_results and
                color_analysis_results['confidence'] > 0.7 and
                color_analysis_results['has_shape']):
            final_detections.append({
                'bbox': None,
                'confidence': color_analysis_results['confidence'],
                'class_id': None,
                'class_name': color_analysis_results['class_name'],
                'original_class': None,
                'color_override': True
            })
        elif (low_confidence_mode and
              color_analysis_results and
              color_analysis_results['confidence'] > 0.5 and
              color_analysis_results['has_shape']):
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
    for det in final_detections:
        if det['bbox']:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']
            color = (0, 255, 0) if det['color_override'] else (255, 0, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{class_descriptions[class_name]}: {conf:.2f}"
            if det['color_override']:
                label += " (颜色分析)"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img


def enhance_image_contrast(image):
    """增强图像对比度"""
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
    return enhanced


# 检查模型是否存在
if not os.path.exists(model_path):
    st.sidebar.error(f"❌ 模型文件不存在: {model_path}")
    st.info("请确保模型文件已上传到仓库根目录")
    st.stop()


# 加载模型（带缓存和错误处理）
@st.cache_resource(ttl=3600)
def load_model(model_path):
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"❌ 模型加载失败: {str(e)}")
        st.info("请检查模型文件是否有效")
        return None


model = load_model(model_path)
if model is None:
    st.stop()

# 主界面
st.write("上传图像或使用摄像头进行检测")

# 创建全局输出占位符（关键！避免DOM冲突）
main_output = st.empty()
debug_output = st.empty()

# 选项卡
tab1, tab2 = st.tabs(["🖼️ 图像上传", "📸 摄像头"])

with tab1:
    uploaded_file = st.file_uploader("选择图像文件", type=['jpg', 'jpeg', 'png', 'bmp'], key="image_uploader")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="上传的图像", use_column_width=True)

        if st.button("🔍 检测香蕉成熟度", key="image_detect"):
            with st.spinner("正在检测..."):
                # 保存临时图像
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    image.save(tmp.name)
                    tmp_path = tmp.name

                try:
                    # 读取并预处理图像
                    image_array = cv2.imread(tmp_path)
                    if enhance_contrast:
                        image_array = enhance_image_contrast(image_array)

                    # 预测
                    conf_threshold = 0.3 if low_confidence_mode else confidence_threshold
                    results = model.predict(tmp_path, conf=conf_threshold, verbose=False)

                    # 处理结果
                    final_detections, color_analysis_results = process_detections(
                        results, image_array, use_color_analysis, low_confidence_mode
                    )

                    # 清空并更新主输出
                    with main_output.container():
                        if final_detections:
                            result_img = visualize_results(image_array, final_detections, class_descriptions)
                            st.image(result_img, caption="✅ 检测结果", use_column_width=True)

                            st.subheader("检测详情")
                            for i, det in enumerate(final_detections):
                                if det['bbox']:
                                    st.write(
                                        f"{i + 1}. **{class_descriptions[det['class_name']]}** (置信度: {det['confidence']:.2f})")
                                    if det['color_override']:
                                        st.caption(
                                            f"→ 原始YOLO结果: {class_descriptions[det['original_class']]} (已被颜色分析覆盖)")
                                else:
                                    st.write(
                                        f"{i + 1}. **{class_descriptions[det['class_name']]}** (仅颜色分析, 置信度: {det['confidence']:.2f})")

                            # 颜色分析图表
                            if use_color_analysis and color_analysis_results:
                                features = extract_color_features(image_array)
                                fig = create_color_analysis_chart(image_array, features)
                                st.pyplot(fig, clear_figure=True)

                                with st.expander("📊 颜色特征详情", expanded=False):
                                    st.write(f"- 黄色区域比例: {features['yellow_ratio']:.2%}")
                                    st.write(f"- 绿色区域比例: {features['green_ratio']:.2%}")
                                    st.write(f"- 棕色区域比例: {features['brown_ratio']:.2%}")
                        else:
                            st.error("❌ 未检测到香蕉")
                            if show_debug_info:
                                with debug_output.container():
                                    st.warning("开启调试模式查看详细信息")
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

with tab2:
    st.write("使用摄像头进行实时检测")
    st.info("💡 **使用提示**: 确保香蕉清晰可见、光线充足、占据画面主要部分")

    camera_image = st.camera_input("拍照", key="camera_input")

    if camera_image is not None:
        image = Image.open(camera_image)
        st.image(image, caption="摄像头图像", use_column_width=True)

        if st.button("🔍 检测香蕉成熟度", key="camera_detect"):
            with st.spinner("正在检测..."):
                try:
                    # 转换图像
                    image_np = np.array(image)
                    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                    # 预处理
                    processed_image = enhance_image_contrast(image_np) if enhance_contrast else image_np.copy()

                    # 调整尺寸（如果太大）
                    if max(processed_image.shape[:2]) > 1280:
                        scale = 1280 / max(processed_image.shape[:2])
                        new_size = (int(processed_image.shape[1] * scale), int(processed_image.shape[0] * scale))
                        processed_image = cv2.resize(processed_image, new_size)

                    # 预测
                    conf_threshold = 0.3 if low_confidence_mode else confidence_threshold
                    results = model.predict(processed_image, conf=conf_threshold, verbose=False)
                    final_detections, color_analysis_results = process_detections(
                        results, processed_image, use_color_analysis, low_confidence_mode
                    )

                    # 如果第一次没检测到，尝试更低阈值
                    if not final_detections:
                        results2 = model.predict(processed_image, conf=0.2, verbose=False)
                        final_detections, color_analysis_results = process_detections(
                            results2, processed_image, use_color_analysis, low_confidence_mode
                        )

                    # 更新主输出
                    with main_output.container():
                        if final_detections:
                            # 调整坐标回原始尺寸（如果需要）
                            if processed_image.shape != image_np.shape:
                                scale_h = image_np.shape[0] / processed_image.shape[0]
                                scale_w = image_np.shape[1] / processed_image.shape[1]
                                for det in final_detections:
                                    if det['bbox']:
                                        x1, y1, x2, y2 = det['bbox']
                                        det['bbox'] = [
                                            int(x1 * scale_w),
                                            int(y1 * scale_h),
                                            int(x2 * scale_w),
                                            int(y2 * scale_h)
                                        ]

                            result_img = visualize_results(image_np, final_detections, class_descriptions)
                            st.image(result_img, caption="✅ 检测结果", use_column_width=True)

                            st.success("检测成功！")
                            for i, det in enumerate(final_detections):
                                if det['bbox']:
                                    st.write(
                                        f"{i + 1}. **{class_descriptions[det['class_name']]}** (置信度: {det['confidence']:.2f})")
                                    if det['color_override']:
                                        st.caption(
                                            f"→ 原始YOLO结果: {class_descriptions[det['original_class']]} (已被颜色分析覆盖)")
                                else:
                                    st.write(
                                        f"{i + 1}. **{class_descriptions[det['class_name']]}** (仅颜色分析, 置信度: {det['confidence']:.2f})")

                            if use_color_analysis and color_analysis_results:
                                features = extract_color_features(processed_image)
                                fig = create_color_analysis_chart(processed_image, features)
                                st.pyplot(fig, clear_figure=True)
                        else:
                            st.error("❌ 未检测到香蕉")
                            if show_debug_info:
                                with debug_output.container():
                                    st.warning("开启调试模式查看详细信息")
                except Exception as e:
                    st.error(f"检测过程中出错: {str(e)}")

# 页脚
st.markdown("---")
st.caption("使用YOLOv8模型和颜色特征分析训练的香蕉成熟度检测系统")