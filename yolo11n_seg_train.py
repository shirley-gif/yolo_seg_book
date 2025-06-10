'''
Segment bookspine images before ocr
'''

from ultralytics import YOLO

# Load a model
#model = YOLO("yolo11m-seg.yaml")  # build a new model from YAML
model = YOLO("yolo11m-seg.pt")  # load a pretrained model (recommended for training)
#model = YOLO("yolo11m-seg.yaml").load("yolo11m.pt")  # build from YAML and transfer weights

# Train the model
#results = model.train(data="coco8-seg.yaml", epochs=100, imgsz=640)
#results = model.train(data="Bookshlef_detection.v5i.yolov11/data.yaml", epochs=100, imgsz=640)

results = model.train(
    data="Bookshlef_detection.v5i.yolov11/data.yaml",
    epochs=100,
    imgsz=640,
    batch=4,     # 建议batch小一些（如2~4），避免梯度不稳定
    device="cpu" ,
    name="bookspine-seg"
)