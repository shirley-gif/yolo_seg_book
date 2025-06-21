'''
Export complete (3.0s)
Results saved to /mnt/d/VSCodeProjects/yolo_seg_book/runs/segment/bookspine-seg-resume/weights
Predict:         yolo predict task=segment model=runs/segment/bookspine-seg-resume/weights/best.onnx imgsz=640  
Validate:        yolo val task=segment model=runs/segment/bookspine-seg-resume/weights/best.onnx imgsz=640 data=book_datasets/Book_spine_2.v4i.yolov11/data.yaml  
Visualize:       https://netron.app
'''

from ultralytics import YOLO

# 加载训练好的 YOLOv8 分割模型
model = YOLO("runs/segment/bookspine-seg-resume/weights/best.pt")

# 导出为 ONNX 格式
model.export(format="onnx", opset=12, dynamic=True)