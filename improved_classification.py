import os
import shutil
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
from ultralytics import YOLO
import matplotlib.pyplot as plt


def extract_color_features(image):
    """提取图像的颜色特征"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_hsv = np.mean(hsv, axis=(0, 1))

    # 黄色区域
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_ratio = np.sum(yellow_mask > 0) / (image.shape[0] * image.shape[1])

    # 绿色区域
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.sum(green_mask > 0) / (image.shape[0] * image.shape[1])

    # 棕色区域
    lower_brown = np.array([8, 60, 20])
    upper_brown = np.array([20, 255, 200])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    brown_ratio = np.sum(brown_mask > 0) / (image.shape[0] * image.shape[1])

    # 亮度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness_mean = np.mean(gray)
    brightness_std = np.std(gray)

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


def safe_classify_by_color_features(image_array):
    """安全版本的分类，避免无香蕉场景的误判"""
    if image_array is None:
        return "no_banana", 0.0, False

    # 检查是否有香蕉形状
    has_shape = contains_banana_shape(image_array)

    # 提取颜色特征
    features = extract_color_features(image_array)

    # 计算总香蕉颜色比例
    total_banana_color = features['yellow_ratio'] + features['green_ratio'] + features['brown_ratio']

    # 安全阈值：如果没有形状且颜色比例很低，直接判定无香蕉
    if not has_shape and total_banana_color < 0.3:
        return "no_banana", 0.0, has_shape

    # 如果有形状或颜色比例高，继续分类
    # 调整置信度：有形状的置信度更高
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
    elif features['yellow_ratio'] > 0.5:
        if total_banana_color > 0.6:
            confidence = min(0.9, features['yellow_ratio'] * 2 * shape_boost)
            return "ripe", confidence, has_shape
        else:
            return "no_banana", 0.0, has_shape
    else:
        return "no_banana", 0.0, has_shape


def improved_classify_test_images_safe(test_dir="test", output_dir="safe_classification_results",
                                       model_path="banana_detection_yolov8_final.pt"):
    """
    使用安全颜色分析分类图片
    """
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        return

    # 创建输出目录
    categories = ["overripe", "ripe", "rotten", "unripe", "no_banana"]
    for category in categories:
        os.makedirs(os.path.join(output_dir, category), exist_ok=True)

    # 加载模型
    print("正在加载模型...")
    model = YOLO(model_path)
    class_names = model.names

    # 处理图片
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(test_dir) if any(f.lower().endswith(ext) for ext in image_extensions)]

    stats = defaultdict(int)
    results_data = []

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(test_dir, image_file)
        print(f"处理图片 {i + 1}/{len(image_files)}: {image_file}")

        # 读取图像
        image_array = cv2.imread(image_path)
        if image_array is None:
            print(f"  无法读取图片: {image_file}")
            continue

        # YOLO检测
        results = model(image_path, verbose=False)

        detected_classes = []
        confidences = []

        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                detected_classes.extend(boxes.cls.tolist())
                confidences.extend(boxes.conf.tolist())

        # 安全颜色分析
        color_class, color_confidence, has_shape = safe_classify_by_color_features(image_array)

        # 决策逻辑
        if detected_classes:
            # YOLO检测到
            max_conf_idx = np.argmax(confidences)
            yolo_class = class_names[int(detected_classes[max_conf_idx])]
            yolo_confidence = confidences[max_conf_idx]

            # 如果颜色分析也检测到香蕉且有形状，可以谨慎覆盖
            if (color_class != "no_banana" and
                    has_shape and
                    color_confidence > 0.6 and
                    abs(yolo_confidence - color_confidence) < 0.2):
                final_class = color_class
                final_confidence = (yolo_confidence + color_confidence) / 2
                print(f"  -> 安全覆盖: {color_class} (有形状验证)")
            else:
                final_class = yolo_class
                final_confidence = yolo_confidence
                print(f"  -> YOLO结果: {yolo_class}")
        else:
            # YOLO未检测到
            if color_class != "no_banana" and has_shape and color_confidence > 0.6:
                final_class = color_class
                final_confidence = color_confidence
                print(f"  -> 颜色分析+形状验证: {color_class}")
            else:
                final_class = "no_banana"
                final_confidence = 0.0
                print(f"  -> 未检测到香蕉")

        # 保存结果
        dest_path = os.path.join(output_dir, final_class, image_file)
        shutil.copy2(image_path, dest_path)

        stats[final_class] += 1
        results_data.append({
            'image': image_file,
            'predicted_class': final_class,
            'confidence': final_confidence,
            'has_shape': has_shape,
            'color_class': color_class,
            'color_confidence': color_confidence,
            'yolo_detected': len(detected_classes) > 0
        })

    # 保存结果
    results_df = pd.DataFrame(results_data)
    csv_path = os.path.join(output_dir, "safe_classification_results.csv")
    results_df.to_csv(csv_path, index=False)

    print(f"\n安全分类统计:")
    for category in categories:
        print(f"  {category}: {stats[category]} 张图片")

    return results_df


if __name__ == "__main__":
    # 使用安全版本进行分类
    results = improved_classify_test_images_safe()