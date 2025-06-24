import cv2
import numpy as np

# 示例：模拟一些预测框和掩膜信息
# 模拟预测框（x1, y1, x2, y2）
boxes = np.array([
    [100, 150, 200, 300],
    [105, 155, 205, 305],  # 与上一个框几乎重复
    [400, 100, 500, 250]   # 独立一个
])

# 模拟掩膜区域（640x640 二值图）
masks = np.zeros((3, 640, 640), dtype=np.uint8)
masks[0, 150:300, 100:200] = 255
masks[1, 155:305, 105:205] = 255
masks[2, 100:250, 400:500] = 255

# 方法：用 NMS 合并重复框
def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """简易 NMS 实现"""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

# 模拟置信度
scores = np.array([0.95, 0.9, 0.85])

# 过滤重复框
keep_indices = non_max_suppression(boxes, scores, iou_threshold=0.4)

# 结果：只保留非重复框对应的掩膜和框
filtered_boxes = boxes[keep_indices]
filtered_masks = masks[keep_indices]

print(filtered_boxes, filtered_masks.shape)
