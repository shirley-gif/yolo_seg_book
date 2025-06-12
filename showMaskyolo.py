import cv2
import numpy as np
from ultralytics import YOLO

# 载入模型和图片
model = YOLO("runs/segment/bookspine-seg4/weights/best.pt")
img_path = "dku_data/Weixin Image_20250611162226.jpg"  # 换成你实际图片路径
img = cv2.imread(img_path)

# 推理得到结果
results = model(img_path)
result = results[0]

if result.masks is not None:
    for i, mask in enumerate(result.masks.data.cpu().numpy()):
        # 1. 将掩码转为0~255的灰度图
        mask_gray = (mask * 255).astype(np.uint8)

        # 2. 调整掩码大小到和原图一致
        mask_resized = cv2.resize(mask_gray, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 3. 显示原图和掩码
        cv2.imwrite("mask_resized.jpg", mask_resized)

        # 4. 将掩码应用到原图做半透明蒙版叠加，显示效果
        color = np.array([0, 255, 0], dtype=np.uint8)  # 绿色蒙版
        overlay = img.copy()
        overlay[mask_resized > 127] = overlay[mask_resized > 127] * 0.4 + color * 0.6
        #cv2.imshow(f"Overlay with Mask {i+1}", overlay)
        cv2.imwrite("output_overlay.jpg", overlay)

        cv2.waitKey(0)
cv2.destroyAllWindows()
