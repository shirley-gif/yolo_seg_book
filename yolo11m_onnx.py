'''
安装依赖：pip install onnxruntime opencv-python numpy
使用训练好的ONNX模型，进行推理


output[0]: shape (1, 37, 8400)   ← 预测框（boxes + mask 系数）
output[1]: shape (1, 32, 160, 160) ← 掩膜原型（mask prototype）


output[0]:预测框+掩膜系数
(1,37,8400)

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
mask_coeff = outputs[0][0][6:]


protos = outputs[1]

for i in range(8400):
    if outputs[0][0][4,i]>0.8:

        # 假设 i 是你感兴趣的目标
        coeffs = outputs[0][0][5:38, i]  # 取出 32 个 mask 系数，shape: (32,)

        # 线性组合
        pred_mask = np.sum(coeffs[:, None, None] * proto, axis=0)  # (160, 160)

        # sigmoid 激活
        pred_mask = 1 / (1 + np.exp(-pred_mask))  # normalize to 0–1
        
           # 二值化阈值0.5
        mask_bin = (pred_mask > 0.5).astype(np.uint8) * 255

         # 放大到640x640，模拟原图大小
        mask_up = cv2.resize(mask_bin, (640, 640), interpolation=cv2.INTER_NEAREST)


        # 可视化
        plt.figure(figsize=(12,6))
        for j in range(1,10):
            plt.subplot(2,6,j)
            plt.title("MASK PROTOTYPE (No. 1)")
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

        plt.show()

        break



