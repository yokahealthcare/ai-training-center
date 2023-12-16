from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s-pose.pt")  # load a pretrained model (recommended for training)
success = model.export(format="onnx", opset=11, simplify=True)  # export the model to onnx format
assert success

# / Tobe continued
