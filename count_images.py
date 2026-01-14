import os
import glob
from collections import defaultdict

def count_images_in_directory(directory_path):
    """
    统计指定目录及其子目录中的图片数量
    
    Args:
        directory_path (str): 要统计的目录路径
        
    Returns:
        dict: 包含各目录图片数量的字典
    """
    # 支持的图片扩展名
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff', '*.webp']
    
    # 存储结果
    results = defaultdict(int)
    total_count = 0
    
    # 遍历目录
    for root, dirs, files in os.walk(directory_path):
        # 统计当前目录中的图片数量
        image_count = 0
        for ext in image_extensions:
            # 使用glob模式匹配
            pattern = os.path.join(root, ext)
            image_count += len(glob.glob(pattern))
            # 也匹配大写扩展名
            pattern_upper = os.path.join(root, ext.upper())
            image_count += len(glob.glob(pattern_upper))
        
        if image_count > 0:
            # 获取相对路径
            relative_path = os.path.relpath(root, directory_path)
            results[relative_path] = image_count
            total_count += image_count
    
    return results, total_count

def print_results(results, total_count):
    """
    打印统计结果
    
    Args:
        results (dict): 包含各目录图片数量的字典
        total_count (int): 总图片数量
    """
    print("=" * 50)
    print("图片数量统计结果")
    print("=" * 50)
    
    # 按目录层级排序并打印
    for path, count in sorted(results.items()):
        print(f"{path}: {count} 张图片")
    
    print("-" * 50)
    print(f"总计: {total_count} 张图片")
    print("=" * 50)

if __name__ == "__main__":
    # 设置要统计的目录路径
    archive_path = os.path.join(os.path.dirname(__file__), "archive")
    
    if not os.path.exists(archive_path):
        print(f"错误: 目录 {archive_path} 不存在")
    else:
        results, total_count = count_images_in_directory(archive_path)
        print_results(results, total_count)