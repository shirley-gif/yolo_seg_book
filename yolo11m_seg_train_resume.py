from ultralytics import YOLO

# 加载你训练过的模型
model = YOLO("runs/segment/bookspine-seg/weights/best.pt")

# 设置训练配置
train_args = {
    "data": "book_datasets/Book_spine_2.v4i.yolov11/data.yaml",
    "imgsz": 640,
    "batch": 4,
    "device": 0,
    "epochs": 1,
    "save": False,
    "verbose": False,
    "name": "bookspine-seg-resume",  # 每次都写死
    "exist_ok": True                # 不会自动改成 seg-resume2、seg-resume3
}

# 早停参数
total_epochs = 50
patience = 5
best_map = 0
epochs_no_improve = 0

# 开始循环训练
for epoch in range(total_epochs):
    print(f"Epoch {epoch+1}/{total_epochs}")
    results = model.train(**train_args)
    # 获取 mAP50-95
    try:
        metric = results.results_dict['metrics/mAP50-95(M)']
        print(metric)
        current_map = metric
        
    except Exception as e:
        print("Warning: Failed to read mAP50-95 from results:", e)
        current_map = 0
    print("current_map : ",current_map)
    print("best_map:", best_map)
    if current_map > best_map:
        best_map = current_map
        epochs_no_improve = 0
        model.save(f"best_map_epoch_{epoch+1}.pt")
        print(f"✅ New best mAP50-95: {current_map:.4f}, model saved.")
    else:
        epochs_no_improve += 1
        print(f"mAP50-95 did not improve ({current_map:.4f}). No improvement for {epochs_no_improve} epochs.")

    if epochs_no_improve >= patience:
        print(f"⏹ Early stopping triggered at epoch {epoch+1}")
        break
