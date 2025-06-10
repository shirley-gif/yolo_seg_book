from ultralytics import YOLO
import glob
import os
import numpy as np
import cv2
import random

# Load a model
#model = YOLO("yolo11m-seg.pt")  # load an official model
model = YOLO("runs/segment/bookspine-seg5/weights/best.pt")  # load a custom model
#("runs/segment/train/weights/best.pt")  # load a custom model
#model = YOLO("runs/segment/bookspine-seg/weights/best.pt")
img_paths = glob.glob('bookspine/images/test/*.jpg') # all the images waiting to be tested 
results_summary = []



for img_path in img_paths: 
    img = cv2.imread(img_path)
    results = model(img_path)
    result = results[0]
    num_books = 0
    if result.masks is not None:
        num_books = len(result.masks.data) # the number of books detected
        for i, mask in enumerate(result.masks.data.cpu().numpy()):
            # 获取每个mask的外接矩形
            mask_bin = (mask * 255).astype(np.uint8)
            mask_resized = cv2.resize(mask_bin, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # --- 新增：生成彩色mask ---
            color = [random.randint(0,255) for _ in range(3)]
            img[mask_resized > 127] = img[mask_resized > 127] * 0.4 + np.array(color) * 0.6   # 半透明叠加

            
            if len(contours) > 0:
                x, y, w, h = cv2.boundingRect(contours[0])
                # 画矩形
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=1)  # 颜色红，线宽3
                # 写文字编号
                font_scale = 1   # 字体大小
                thickness = 2   # 字体粗细
                text = f'Book {i+1}'
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                text_x = x
                text_y = max(y - 10, text_size[1] + 10)
                cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
    #print(f'{img_path}: Detected {num_books} book spines.')
    
    results_summary.append({
        'image': os.path.basename(img_path),
        'num_books_segmented': num_books
    })

    # ----- Core change: save under segmented subfolder -----
    original_dir = os.path.dirname(img_path)
    save_dir = os.path.join(original_dir, "segmented")
    os.makedirs(save_dir, exist_ok=True)   # create if not exists
    save_path = os.path.join(save_dir, os.path.basename(img_path))
    
    # create image based on masks:
    cv2.imwrite(save_path, img)
    #results[0].save(save_path) # save the segmented image 


# 汇总所有图片
print('\nSummary:')
for r in results_summary:
    print(f"{r['image']}: {r['num_books_segmented']} books segmented.")

# Predict with the model
#results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
#results = model("bookspine/images/test/bookspine001.jpg")
# Access the results
#for result in results:
#    xy = result.masks.xy  # mask in polygon format
#    xyn = result.masks.xyn  # normalized
#    masks = result.masks.data  # mask in matrix format (num_objects x H x W)

#results[0].show()  # 直接弹窗可视化
