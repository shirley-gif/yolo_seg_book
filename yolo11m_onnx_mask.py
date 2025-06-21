'''
安装依赖：pip install onnxruntime opencv-python numpy
使用训练好的ONNX模型，进行推理


output[0]: shape (1, 37, 8400)   ← 预测框（boxes + mask 系数）
output[1]: shape (1, 32, 160, 160) ← 掩膜原型（mask prototype）


output[0]:预测框+掩膜系数
(1,37,8400)

画出一个预测框的掩膜


'''

import onnxruntime as ort
import numpy as np
import cv2

import matplotlib.pyplot as plt

#from understandProtoMask import cal_mask

# 加载图像并预处理
image = cv2.imread("dku_data/xie.jpg")
img = cv2.resize(image, (640, 640))
img = img[:, :, ::-1]  # BGR → RGB
img = img.transpose(2, 0, 1).astype(np.float32) / 255.0  # CHW, normalize
img = np.expand_dims(img, axis=0)  # [1, 3, 640, 640]

# 使用 CPU 加载 ONNX 模型
session = ort.InferenceSession("runs/segment/bookspine-seg-resume/weights/best.onnx", providers=["CPUExecutionProvider"])

# 获取模型输入名
input_name = session.get_inputs()[0].name


# 推理
outputs = session.run(None, {input_name: img})

# outputs 结构通常为 [boxes, masks, ...]
print("输出 tensor 数量:", len(outputs))
print("每个输出 shape:")
for o in outputs:
    print(np.array(o).shape)

proto = outputs[1][0]          # shape: (32, 160, 160)
preds = outputs[0][0]       # shape: (37, 8400)
# preds[4] 是置信度，preds[5:37] 是 mask 系数
# 处理每个预测框
for i in range(8400):
    if preds[4,i]>0.7:

        # 假设 i 是你感兴趣的目标
        coeffs = preds[5:37, i]  # 取出 32 个 mask 系数，shape: (32,)
        print("coeffs:", coeffs[:5])  # 看看是不是都接近0或全负

        # 线性组合
        pred_mask = np.sum(coeffs[:, None, None] * proto, axis=0)  # (160, 160)

        # sigmoid 激活
        pred_mask = 1 / (1 + np.exp(-pred_mask))  # normalize to 0–1
        
        # 二值化阈值0.5
        mask_bin = (pred_mask > 0.3).astype(np.uint8) * 255

         # 放大到640x640，模拟原图大小
        mask_up = cv2.resize(mask_bin, (640, 640), interpolation=cv2.INTER_NEAREST)
        print(f"cx={preds[0, i]}, cy={preds[1, i]}, w={preds[2, i]}, h={preds[3, i]}")

        x_center = preds[0, i] 
        y_center = preds[1, i] 
        w = preds[2, i] 
        h = preds[3, i] 

        x1 = max(0, int(x_center - w / 2))
        y1 = max(0, int(y_center - h / 2))
        x2 = min(640, int(x_center + w / 2))
        y2 = min(640, int(y_center + h / 2))

        if x2 <= x1 or y2 <= y1:
            print(f"⚠️ 无效 bbox: x1={x1}, x2={x2}, y1={y1}, y2={y2}")
            continue

        mask_crop = mask_up[y1:y2, x1:x2]
        print(mask_crop)
        
        # 原图对应区域
        roi = image[y1:y2, x1:x2]

        # 创建彩色遮罩（绿色为例）
        color_mask = np.zeros_like(roi)
        color_mask[mask_crop > 0] = [0, 255, 0]  # BGR: Green

        # 将遮罩与原图 ROI 融合（透明度 50%）
        blended = cv2.addWeighted(roi, 0.5, color_mask, 0.5, 0)

        # 替换回原图
        image[y1:y2, x1:x2] = blended
        
        # 可视化
        plt.figure(figsize=(24,12))
        for j in range(1,9):
            plt.subplot(2,6,j)
            plt.title("PROTO (No. {})".format(j))   
            plt.imshow(proto[j], cmap='gray')
            plt.axis('off')

        plt.subplot(2,6,11)
        plt.title("MASK AFTER LINEAR COEFFICIENT (160x160)")
        plt.imshow(pred_mask, cmap='gray')
        plt.axis('off')

        plt.subplot(2,6,12)
        plt.title("onnx MASK AFTER RESIZE (640x640)")
        plt.imshow(mask_up, cmap='gray')
        plt.axis('off')

        plt.subplot(2,6,10)
        plt.title("onnx MASK CROP (640x640)")
        # plt.imshow(mask_crop, cmap='gray')
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

        break



