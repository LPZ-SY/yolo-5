import os
folder = "D:\phd-project\dataset\PathText\datasets"
final_dict = {}
for sub_folder in os.listdir(folder):
    print(f"{sub_folder} : {len(os.listdir(os.path.join(folder, sub_folder)))}")
    for name in os.listdir(os.path.join(folder, sub_folder)):
        final_dict[name] = sub_folder

print(final_dict)

# 然后看下目前，我已经有的特征基本都是占有多少个
nums_get = {}
for x in os.listdir("H:\medical_data\GDC_Data\Just_clam\FEATURES_DIRECTORY\pt_files"):
    x_names = x[:12]
    print(x_names)
    try:
        cls = final_dict[x_names]
        if cls in nums_get.keys():
            nums_get[cls] = nums_get[cls] + 1
        else:
            nums_get[cls] = 1
    except:
        print("拉了")

print(nums_get)

# TCGA-KIRC': 238 （kidney）
# 'TCGA-LGG': 181, （brain）
# 'TCGA-BRCA': 104, （breast）
# 'TCGA-COAD': 148, 这个有两个类别 暂时不考虑