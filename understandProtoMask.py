'''
┌───────────────────────┐
│       输入图像 (640x640)      │
└─────────────┬─────────┘
              │
              ▼
      Backbone网络提取特征
              │
              ├─────────────┐
              │             │
              ▼             ▼
(1) 原型掩膜网络输出 (Proto Masks)
   proto shape: [32, 160, 160]  ← 一组“模板掩膜”
              │
              │
              ▼
(2) 探测头输出
    每个目标预测 mask_coeff (32维向量)
    还有box和类别信息
              │
              │
              ▼
(3) 对每个目标掩膜进行线性组合：
    mask = sum_{i=1}^{32} mask_coeff[i] * proto[i]
              │
              ▼
(4) mask经过 sigmoid，二值化后resize回原图大小
              │
              ▼
(5) 最终目标掩膜 (和对应box一起用作分割结果)

'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# 假设 proto 是模型输出的原型掩膜，shape=[32,160,160]
# mask_coeff 是某个目标的系数，shape=[32]

# 这里用随机数据模拟
proto = np.random.rand(32, 160, 160).astype(np.float32)
mask_coeff = np.random.randn(32).astype(np.float32)


def cal_mask(proto, mask_coeff):

    # 线性组合得到目标掩膜
    # mask = sum_i coeff[i] * proto[i]
    mask = np.tensordot(mask_coeff, proto, axes=1)  # shape = (160,160)

    # 经过sigmoid归一化
    mask = 1 / (1 + np.exp(-mask))

    # 二值化阈值0.5
    mask_bin = (mask > 0.5).astype(np.uint8) * 255

    # 放大到640x640，模拟原图大小
    mask_up = cv2.resize(mask_bin, (640, 640), interpolation=cv2.INTER_NEAREST)

    return mask_up, mask

mask_up, mask = cal_mask(proto, mask_coeff)
# 可视化
plt.figure(figsize=(12,6))
plt.subplot(1,3,1)
plt.title("MASK PROTOTYPE (No. 1)")
plt.imshow(proto[0], cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)
plt.title("MASK AFTER LINEAR COEFFICIENT WITH PROTOTYPE (160x160)")
plt.imshow(mask, cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title("MASK AFTER RESIZE (640x640)")
plt.imshow(mask_up, cmap='gray')
plt.axis('off')

plt.show()
