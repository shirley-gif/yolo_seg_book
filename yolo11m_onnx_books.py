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

# 计算出每个预测框的掩膜
proto = outputs[1][0]          # shape: (32, 160, 160)
preds = outputs[0][0]       # shape: (37, 8400)
# preds[4] 是置信度，preds[5:37] 是 mask 系数
# 处理每个预测框    

conf = preds[4]
indices = np.where(conf > 0.85)[0]
print(f"检测到 {len(indices)} 个高置信度目标")

for i in indices:
    print(f"处理第 {i} 个预测框")
    # 取出每个预测框的 mask 系数
    # preds 的 shape 是 (37, 8400)，其中第 5 到 36 行是 mask 系数
    # 这里的 i 是预测框的索引
    # preds[5:37, i] 取出第 i 个预测框的 mask 系数，shape: (32,)
    # 这些系数是模型为每个预定义的掩膜原型分配的权重
    # 这些系数应该是接近 0 或全负的，因为它们表示每个掩膜原型对当前预测框的贡献程度
    # 如果系数接近 0 或全负，说明当前预测框与这些掩膜原型的匹配度较低
    # 如果系数较大，说明当前预测框与这些掩膜原型的匹配度较高
    # 这里的 preds[5:37, i] 是一个 32 维    
    coeffs = preds[5:37, i]  # 取出 32 个 mask 系数，shape: (32,)
    print("coeffs:", coeffs[:5])  # 看看是不是都接近0或全负



    x_center = preds[0, i] 
    y_center = preds[1, i] 
    w = preds[2, i] 
    h = preds[3, i] 

    x1 = max(0, int(x_center - w / 2))
    y1 = max(0, int(y_center - h / 2))
    x2 = min(640, int(x_center + w / 2))
    y2 = min(640, int(y_center + h / 2))

    # 获取边界框坐标
    #x1,y1,x2,y2 = preds[:4, i]  # 取出边界框坐标，shape: (4,)
    #print(f"边界框坐标: ({x1}, {y1}), ({x2}, {y2})")   
    ## 将它们转为整数并排序，保证 x1 < x2，y1 < y2
    #x1, x2 = sorted([int(x1), int(x2)])
    #y1, y2 = sorted([int(y1), int(y2)])

    print(f"修正后边界框坐标: ({x1}, {y1}) → ({x2}, {y2})") 

    # 线性组合 把系数和 proto 进行线性组合
    # proto 的 shape 是 (32, 160, 160)，coeffs 的 shape 是 (32,)    
    # 结果的 shape 是 (160, 160)
    # 这里的 coeffs 是 mask 系数，proto 是预定义的掩膜原型
    # 线性组合  
    # coeffs[:, None, None] 是 (32, 1, 1)，proto 是 (32, 160, 160)
    # 所以 np.sum(coeffs[:, None, None] * proto, axis=0) 会得到一个 (160, 160) 的掩膜
    # 这一步是将每个 mask 原型与对应的系数相乘      
    # 然后对所有 mask 原型进行加权求和，得到最终的掩膜
    pred_mask = np.sum(coeffs[:, None, None] * proto, axis=0)  # (160, 160)

    # sigmoid 激活  
    # 这里的 pred_mask 可能是一个未归一化的掩膜
    # 使用 sigmoid 函数将其归一化到 0-1 范围
    pred_mask = 1 / (1 + np.exp(-pred_mask))  # normalize to 0–1

    # 将掩膜应用到原图上
    # 这里的 pred_mask 是一个 (160, 160) 的掩膜 
    # 需要将其缩放到原图大小 (640, 640)
    pred_mask_resized = cv2.resize(pred_mask, (640, 640))

    print("pred_mask min/max:", pred_mask.min(), pred_mask.max())
    print("pred_mask_resized > 0.5 count:", np.sum(pred_mask_resized > 0.5))


    # 二值化：掩膜值 > 阈值的为前景
    binary_mask = (pred_mask_resized > 0.5).astype(np.uint8)  # (0或1)

   
    cv2.imshow(f"Binary mask resized {i}", binary_mask*255)
    # 只保留边界框内的掩膜部分，其他设为0
    masked_area = np.zeros_like(binary_mask)

    masked_area[y1:y2, x1:x2] = binary_mask[y1:y2, x1:x2]
    print("掩膜裁剪区域 shape:", binary_mask[y1:y2, x1:x2].shape)

    # 将掩膜转换为伪彩色图像
    masked_area_color = (masked_area * 255).astype(np.uint8)  # 转换为 0-255 范围 
    masked_area_color = cv2.applyColorMap(masked_area_color, cv2.COLORMAP_JET)  # 应用伪彩色映射
    print(img.shape, masked_area_color.shape)

    cv2.imshow(f"Masked area color {i}", masked_area_color)

    #尝试得到BINARY MASK
    color_mask = cv2.applyColorMap((binary_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)


    print("按空格键关闭图像窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    # 将掩膜叠加到原图上
    overlayed_image = cv2.addWeighted(cv2.resize(image, (640, 640)), 0.5, color_mask , 0.5, 0)
    # 画边界框
    cv2.rectangle(overlayed_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 蓝色边框

    # 画掩膜轮廓
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlayed_image, contours, -1, (0, 255, 0), 2)  # 绿色轮廓

    cv2.imshow(f"Overlayed Image with Contours {i}", overlayed_image)

    print("按空格键关闭图像窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
