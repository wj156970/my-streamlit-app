import os
import shutil
import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import pandas as pd
from datetime import datetime


def classify_test_images(test_dir="test", output_dir="classified_results",
                         model_path="banana_detection_yolov8_final.pt"):
    """
    分类test文件夹中的图片，根据香蕉成熟度将图片复制到对应的子文件夹

    参数:
        test_dir: 测试图片目录
        output_dir: 分类结果输出目录
        model_path: 训练好的模型路径
    """

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        print("请确保模型文件存在或提供正确的模型路径")
        return

    # 创建输出目录结构
    categories = ["overripe", "ripe", "rotten", "unripe", "no_banana"]
    for category in categories:
        os.makedirs(os.path.join(output_dir, category), exist_ok=True)

    # 加载模型
    print("正在加载模型...")
    model = YOLO(model_path)

    # 获取类别名称
    class_names = model.names
    print(f"模型类别: {class_names}")

    # 统计结果
    stats = defaultdict(int)
    results_data = []

    # 处理测试目录中的所有图片
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = [f for f in os.listdir(test_dir) if any(f.lower().endswith(ext) for ext in image_extensions)]

    print(f"在 {test_dir} 中找到 {len(image_files)} 张图片")

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(test_dir, image_file)
        print(f"处理图片 {i + 1}/{len(image_files)}: {image_file}")

        # 进行预测
        results = model(image_path, verbose=False)

        # 解析结果
        detected_classes = []
        confidences = []

        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                # 获取检测到的类别和置信度
                detected_classes.extend(boxes.cls.tolist())
                confidences.extend(boxes.conf.tolist())

        # 确定图片的主要类别
        if detected_classes:
            # 如果检测到香蕉，选择置信度最高的类别
            max_conf_idx = np.argmax(confidences)
            main_class_idx = int(detected_classes[max_conf_idx])
            main_class_name = class_names[main_class_idx]
            max_confidence = confidences[max_conf_idx]
        else:
            # 如果没有检测到香蕉
            main_class_name = "no_banana"
            max_confidence = 0.0

        # 复制图片到对应的分类文件夹
        dest_path = os.path.join(output_dir, main_class_name, image_file)
        shutil.copy2(image_path, dest_path)

        # 更新统计
        stats[main_class_name] += 1

        # 记录结果数据
        results_data.append({
            'image': image_file,
            'predicted_class': main_class_name,
            'confidence': max_confidence,
            'detected_objects': len(detected_classes) if detected_classes else 0
        })

        print(f"  -> 分类为: {main_class_name} (置信度: {max_confidence:.2f})")

    # 保存分类结果到CSV
    results_df = pd.DataFrame(results_data)
    csv_path = os.path.join(output_dir, "classification_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"分类结果已保存到: {csv_path}")

    # 打印统计信息
    print("\n分类统计:")
    for category in categories:
        count = stats[category]
        print(f"  {category}: {count} 张图片")

    # 创建可视化结果
    create_visualization(output_dir, stats)

    print(f"\n分类完成! 结果保存在 {output_dir} 目录中")


def create_visualization(output_dir, stats):
    """创建分类结果的可视化图表"""
    try:
        import matplotlib.pyplot as plt

        # 创建饼图
        plt.figure(figsize=(10, 6))

        # 过滤掉计数为0的类别
        filtered_stats = {k: v for k, v in stats.items() if v > 0}

        # 创建饼图
        plt.pie(
            filtered_stats.values(),
            labels=filtered_stats.keys(),
            autopct='%1.1f%%',
            startangle=90
        )

        plt.title('香蕉成熟度分类结果分布')

        # 保存图表
        chart_path = os.path.join(output_dir, "classification_chart.png")
        plt.savefig(chart_path)
        plt.close()

        print(f"分类结果图表已保存到: {chart_path}")
    except ImportError:
        print("matplotlib未安装，跳过创建可视化图表")
    except Exception as e:
        print(f"创建可视化图表时出错: {e}")


def create_sample_collage(output_dir, samples_per_class=3):
    """创建一个包含每个类别样本的拼贴图"""
    try:
        categories = ["overripe", "ripe", "rotten", "unripe", "no_banana"]
        sample_images = []

        for category in categories:
            category_dir = os.path.join(output_dir, category)
            if os.path.exists(category_dir):
                images = [f for f in os.listdir(category_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))]
                # 随机选择样本
                import random
                selected = random.sample(images, min(samples_per_class, len(images)))
                for img in selected:
                    img_path = os.path.join(category_dir, img)
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (200, 200))
                    # 添加类别标签
                    cv2.putText(img, category, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    sample_images.append(img)

        if sample_images:
            # 创建拼贴图
            rows = int(np.ceil(len(sample_images) / 4))
            collage = np.zeros((rows * 200, 4 * 200, 3), dtype=np.uint8)

            for i, img in enumerate(sample_images):
                row = i // 4
                col = i % 4
                collage[row * 200:(row + 1) * 200, col * 200:(col + 1) * 200] = img

            # 保存拼贴图
            collage_path = os.path.join(output_dir, "sample_collage.jpg")
            cv2.imwrite(collage_path, collage)
            print(f"样本拼贴图已保存到: {collage_path}")
    except Exception as e:
        print(f"创建样本拼贴图时出错: {e}")


if __name__ == "__main__":
    # 分类test文件夹中的图片
    classify_test_images()

    # 创建样本拼贴图
    create_sample_collage("classified_results")