from ultralytics import YOLO

model = YOLO("car-cctv-6.pt")  # load a custom trained model

# Export the model
model.export(format="onnx")
