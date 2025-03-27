from ultralytics import YOLO

# Load a model
model = YOLO("../42_demo/yolov8n.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom trained model

# Export the model
# model.export(format="onnx", dynamic=True, int8=True)
model.export(format="onnx")

'''
format: 导出模型的目标格式（例如："......"）、 onnx, torchscript, tensorflow).
imgsz: 模型输入所需的图像大小（例如："......"）、 640 或 (height, width)).
half: 启用 FP16 量化，减少模型大小，并可能加快推理速度。
optimize: 针对移动或受限环境进行特定优化。
int8: 启用 INT8 量化，非常有利于边缘部署。
'''

# 安卓文件
# 安卓文件的版本选择2022.2.1