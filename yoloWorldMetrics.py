from ultralytics import YOLOWorld

# Create a YOLO-World model
model = YOLOWorld('yolov8x-worldv2.pt') #YOLO("custom_yolov8s.pt")  # or select yolov8m/l-world.pt for different sizes

model.set_classes(['person with knife with hands exposed'])

# Conduct model validation on the dataset
metrics = model.val(save_json=True,data="Frigo1/yaml.yaml", conf=0.3)




