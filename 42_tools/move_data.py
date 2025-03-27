import os
import os.path as osp
import shutil

src_folder = r"H:\Upppppdate\000-AAAAA-standard-code\yolo11\labelmedata\train"
target_folder = r"H:\Upppppdate\000-AAAAA-standard-code\yolo11\labelmedata\data\jsons"

sub_folders = os.listdir(src_folder)
for sub_folder in sub_folders:
    sub_path = osp.join(src_folder, sub_folder)
    image_names = os.listdir(sub_path)
    for image_name in image_names:
        ima_path = osp.join(sub_path, image_name)
        shutil.copy2(ima_path, target_folder)