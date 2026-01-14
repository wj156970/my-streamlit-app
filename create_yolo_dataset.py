import os
import random
import shutil
from pathlib import Path

def create_yolo_dataset():
    """
    创建YOLO格式的数据集，包括生成标注文件和划分数据集
    """
    # 设置路径
    dataset_root = Path("./archive/Banana Ripeness Classification Dataset")
    train_dir = dataset_root / "train"
    test_dir = dataset_root / "test"
    val_dir = dataset_root / "val"
    
    # 创建验证集目录
    if not val_dir.exists():
        val_dir.mkdir(exist_ok=True)
        print(f"创建验证集目录: {val_dir}")
    
    # 类别映射
    classes = ['overripe', 'ripe', 'rotten', 'unripe']
    
    # 从测试集中移动一部分图像到验证集
    test_classes = [d for d in test_dir.iterdir() if d.is_dir()]
    
    for class_dir in test_classes:
        class_name = class_dir.name
        val_class_dir = val_dir / class_name
        val_class_dir.mkdir(exist_ok=True)
        
        # 获取所有图像文件
        img_files = [f for f in class_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        # 随机选择30%的图像作为验证集
        val_count = max(1, int(len(img_files) * 0.3))
        val_images = random.sample(img_files, val_count)
        
        # 移动选中的图像到验证集
        for img_path in val_images:
            dest_path = val_class_dir / img_path.name
            shutil.move(str(img_path), str(dest_path))
            print(f"移动 {img_path.name} 到验证集")
    
    # 为每个图像创建YOLO格式的标注文件
    for split_dir in [train_dir, val_dir, test_dir]:
        if not split_dir.exists():
            continue
            
        class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        
        for class_dir in class_dirs:
            class_name = class_dir.name
            class_id = classes.index(class_name)
            
            # 为每个图像创建标注文件
            img_files = [f for f in class_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            
            for img_path in img_files:
                # 创建对应的标注文件
                label_path = img_path.with_suffix('.txt')
                
                # 创建一个简单的全图边界框标注
                # 格式: class_id x_center y_center width height
                # 这里我们假设整个图像都是目标区域
                with open(label_path, 'w') as f:
                    f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
                
                print(f"创建标注文件: {label_path}")

if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    random.seed(42)
    
    print("开始创建YOLO格式的数据集...")
    create_yolo_dataset()
    print("数据集创建完成!")