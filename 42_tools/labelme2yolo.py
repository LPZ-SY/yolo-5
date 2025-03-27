import os
import json
import shutil
from sklearn.model_selection import train_test_split
import glob
import yaml

################################### 可选配置项 ###################################################
# 定义原始数据集路径
original_dataset_dir = r"H:\Upppppdate\000-AAAAA-standard-code\yolo11\labelmedata\data"
annotations_dir = os.path.join(original_dataset_dir, 'jsons')  # annotations 目录
images_dir = os.path.join(original_dataset_dir, 'images')  # 图片所在目录
coverted_dataset_dir = r'../hand_new'  # 转换过后的数据集

# 类别到ID的映射，根据实际情况修改
class_map = {
    "1": 0,
    "2": 1,
    "3": 2,
    "4": 3,
    "5": 4,
    "6": 5,
    "7": 6,
    "8": 7,
    "9": 8,
    "A": 9,
    "B": 10,
    "C": 11,
}


################################### 可选配置项 ###################################################

def convert_labelme_to_yolo(json_file, output_label_file):
    # json_str = json_str.replace('\\', '\\\\')
    # db = json.loads(json_str)
    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)

    image_width = data['imageWidth']
    image_height = data['imageHeight']

    with open(output_label_file, 'w') as label_f:
        for shape in data['shapes']:
            label = shape['label']
            points = shape['points']

            # 根据类别名获取类别ID
            class_id = class_map.get(label)
            if class_id is None:
                print(f"警告: 未知类别 {label} 在文件 {json_file}")
                continue

            # YOLO格式要求：class x_center y_center width height
            # 这里的值都是相对于图像宽度和高度的归一化值
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x_center = sum(x_coords) / len(points) / image_width
            y_center = sum(y_coords) / len(points) / image_height
            width = (max(x_coords) - min(x_coords)) / image_width
            height = (max(y_coords) - min(y_coords)) / image_height

            # 写入YOLO格式的标签文件
            label_f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def copy_and_convert(json_files, phase):
    supported_extensions = ['.jpg', '.png']  # 支持的图像格式

    for json_file in json_files:
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        image_copied = False

        for ext in supported_extensions:
            image_file = os.path.join(images_dir, base_name + ext)
            if os.path.isfile(image_file):
                target_image_dir = os.path.join(coverted_dataset_dir, phase, 'images')
                os.makedirs(target_image_dir, exist_ok=True)
                shutil.copy(image_file, target_image_dir)
                image_copied = True
                break

        if not image_copied:
            print(f"警告: 没有找到与 {json_file} 对应的图片文件")
            continue

        target_labels_dir = os.path.join(coverted_dataset_dir, phase, 'labels')
        os.makedirs(target_labels_dir, exist_ok=True)

        output_label_file = os.path.join(target_labels_dir, base_name + '.txt')
        convert_labelme_to_yolo(json_file, output_label_file)


if __name__ == "__main__":
    # 创建转换数据集目录
    if not os.path.exists(coverted_dataset_dir):
        os.mkdir(coverted_dataset_dir)

    # 创建转换数据集目录结构
    for phase in ['train', 'val']:
        for folder in ['images', 'labels']:
            os.makedirs(os.path.join(coverted_dataset_dir, phase, folder), exist_ok=True)

    # 获取所有的json文件
    json_files = glob.glob(os.path.join(annotations_dir, '*.json'))

    # 将数据划分为训练集和验证集
    train_files, val_files = train_test_split(json_files, test_size=0.2, random_state=42)

    # 执行数据集划分与转换
    copy_and_convert(train_files, 'train')
    copy_and_convert(val_files, 'val')

    # 数据集配置信息，相对于 YOLOv5 目录
    data_yaml = {
        'yaml_file_path': os.path.join(coverted_dataset_dir, 'data.yaml'),
        'train': os.path.join('../..', coverted_dataset_dir, 'train', 'images'),
        'val': os.path.join('../..', coverted_dataset_dir, 'val', 'images'),
        'nc': len(class_map),  # 类别数量
        'names': list(class_map.keys())  # 类别名称列表
    }

    # 写入 YAML 文件
    with open(data_yaml['yaml_file_path'], 'w') as f:
        yaml.safe_dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    print(f"YAML 文件已创建：{data_yaml['yaml_file_path']}")
