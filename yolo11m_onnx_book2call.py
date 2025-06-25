'''
Call the onnx model to get the mask for book spine segmentation.
This code is used to predict the book spines in a set of images using an ONNX model.
And call ocr/model to recognize the text on the book spines.
It processes each image, applies the model to detect book spines, and saves the segmented images    
with masks applied.
It also summarizes the number of book spines detected in each image.
'''
import onnxruntime as ort
import numpy as np  
import cv2
import matplotlib.pyplot as plt
import easyocr



def preprocess_image(image_path: str,input_size = (640, 640)) -> np.ndarray:
    """
    Preprocess image for ONNX inference and return scale info.

    Returns:
        img_tensor: (1, 3, H, W) float32
        original_image: original BGR image
        scale: (scale_x, scale_y) for mapping results back
    """
    image = cv2.imread(image_path)
    orig_h, orig_w = image.shape[:2]
    input_w, input_h = input_size

    img = cv2.resize(image, (input_w, input_h))
    scale_x = orig_w / input_w
    scale_y = orig_h / input_h

    img_rgb = img[:, :, ::-1]
    img_tensor = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_tensor = np.expand_dims(img_tensor, axis=0)

    return img_tensor, image, (scale_x, scale_y)

def predict(session, image: np.ndarray) -> np.ndarray:

    """
    Run inference on the preprocessed image using the ONNX model.

    Args:
        image (np.ndarray): Preprocessed image of shape (1, 3, 640, 640).

    Returns:
        np.ndarray: Model output.
    """

    # 获取模型输入名
    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: image})
    return outputs  # Assuming the first output is the desired one

def get_masks(proto, preds, index ):
    '''
    根据预测结果和掩膜原型生成掩膜
    Args:   
        proto: 掩膜原型，shape: (32, 160, 160)
        preds: 模型预测结果，shape: (37, 8400)
        index: 需要处理的预测框索引 
    Returns:
        masked_area: 掩膜区域，shape: (640, 640) 
    '''
    x_center = preds[0, index]
    y_center = preds[1, index]
    w = preds[2, index]
    h = preds[3, index]

    x1 = max(0, int(x_center - w / 2))
    y1 = max(0, int(y_center - h / 2))
    x2 = min(640, int(x_center + w / 2))
    y2 = min(640, int(y_center + h / 2))
    print(f"边界框坐标: ({x1}, {y1}) → ({x2}, {y2})")
    coeffs = preds[5:37, i]  # shape: (32,)   preds[5:37, i] 取出第 i 个预测框的 mask 系数，shape: (32,)
    # coeffs 是一个一维数组，包含 32 个 mask 原型的系数
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

    # 二值化阈值0.5
    binary_mask = (pred_mask_resized > 0.5).astype(np.uint8) * 255 # 转换为 0-255 范围 

    # 只保留边界框内的掩膜部分，其他设为0
    masked_area = np.zeros_like(binary_mask)

    masked_area[y1:y2, x1:x2] = binary_mask[y1:y2, x1:x2]
    print("掩膜裁剪区域 shape:", binary_mask[y1:y2, x1:x2].shape)

    return masked_area

def filter_and_sort_boxes(preds, score_threshold=0.8, nms_threshold=0.4):
    """
    根据置信度阈值和 NMS 筛选预测框，并按从左到右、从上到下排序。

    Args:
        preds: 模型预测输出，shape (37, N)，前 4 行是 bbox，5-36 是掩膜系数
        score_threshold: 置信度筛选阈值
        nms_threshold: NMS 的 IoU 阈值

    Returns:
        keep_indices: 保留的预测框索引（对应 preds 中的索引）
        boxes_scaled: 对应的 (x, y, w, h) 坐标（可用于后处理）
    """
    conf_scores = preds[4]
    selected = np.where(conf_scores > score_threshold)[0]

    boxes = []
    scores = []
    index_map = []

    for idx in selected:
        x_center, y_center, w, h = preds[0:4, idx]
        x = int(x_center - w / 2)
        y = int(y_center - h / 2)
        boxes.append([x, y, int(w), int(h)])
        scores.append(float(conf_scores[idx]))
        index_map.append(idx)

    # Apply OpenCV NMS
    indices_nms = cv2.dnn.NMSBoxes(boxes, scores, score_threshold, nms_threshold)

    if len(indices_nms) == 0:
        return [], []

    # flatten and get original indices
    indices_nms = indices_nms.flatten()
    keep_indices = [index_map[i] for i in indices_nms]
    boxes_kept = [boxes[i] for i in indices_nms]

    # Sort by (x, y) — from left to right, top to bottom
    keep_indices_sorted = [
        idx for _, idx in sorted(
            zip(boxes_kept, keep_indices), key=lambda b: (b[0][1], b[0][0])
        )
    ]
    boxes_sorted = sorted(boxes_kept, key=lambda b: (b[1], b[0]))  # sort by y, then x

    return keep_indices_sorted, boxes_sorted

def get_ocr(img):
    """
    Initialize and return an OCR reader.
    """
    
    reader = easyocr.Reader(['en'], gpu=False)

    #img = cv2.imread('masked_area_8190.png')  # 确保图像路径正确
    results = reader.readtext(img)

    texts = []
    flag = False
    for (box, text, conf) in results:
        #print(f"OCR: {text} （置信度: {conf:.2f}）")
        if text.strip() == 'DKU':
            flag = True
        if flag : 
            texts.append(text)
        box = [tuple(map(int, pt)) for pt in box]
        #cv2.polylines(img, [np.array(box)], True, (0, 255, 0), 2)
        #cv2.putText(img, text, box[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    #text = '_'.join(texts)
    return texts, img

def draw_multiline_text_with_background(
    img,
    text,
    x,
    y,
    h=0,  # 可选：用于整体往下偏移（比如目标框高度的一半）
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.7,
    text_color=(255, 0, 0),
    bg_color=(255, 255, 255),
    thickness=2,
    line_spacing=5
):
    """
    在图像上绘制多行文字，并添加白色背景框。

    参数说明：
    - img: 原图
    - text: 多行文本（使用 \n 分隔）
    - x, y: 左上角起始坐标
    - h: 可选向下偏移量（如目标框高度）
    - 其他为字体和颜色参数
    """
    lines = text
    if not lines or lines == ['']:
        lines = ['(empty)']

    # 单行文字高度
    (_, text_h), baseline = cv2.getTextSize("A", font, font_scale, thickness)
    line_height = text_h + baseline + line_spacing

    # 起始位置
    x0 = int(x)
    y0 = int(y + h / 2)

    # 计算背景框宽高
    max_width = max(cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines)
    box_width = max_width
    box_height = line_height * len(lines) - line_spacing  # 最后一行不需要额外 spacing

    # 背景矩形
    cv2.rectangle(img, (x0, y0), (x0 + box_width, y0 + box_height), bg_color, cv2.FILLED)

    # 逐行写文字
    for i, line in enumerate(lines):
        y_line = y0 + i * line_height + text_h
        cv2.putText(img, line, (x0, y_line), font, font_scale, text_color, thickness)


# BGR to RGB for matplotlib
#img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
_threshold = 0.85  # 置信度阈值
_dimension = (640, 640)  # 模型输入尺寸


# 文本参数
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
thickness = 2
text_color = (255, 0, 0)      # 蓝色
bg_color = (255, 255, 255)    # 白色

# 加载图片，并将它处理为onnx模型输入格式 # Add batch dimension: [1, 3, 640, 640] RGB
img, original_image, scale = preprocess_image("dku_data/dku_barcode.jpg") # Change to your image path

# 使用 CPU 加载 ONNX 模型
session = ort.InferenceSession("runs/segment/bookspine-seg-resume/weights/best.onnx", providers=["CPUExecutionProvider"])

# 进行预测 
outputs = predict(session, img)

# 计算出每个预测框的掩膜
proto = outputs[1][0]       # shape: (32, 160, 160)
preds = outputs[0][0]       # shape: (37, 8400)

# preds[4] 是置信度，preds[5:37] 是 mask 系数
# 处理每个预测框    
conf = preds[4]
indices = np.where(conf > _threshold)[0]
print(f"检测到 {len(indices)} 个高置信度目标")

keep_indices, boxes_sorted = filter_and_sort_boxes(preds, _threshold, 0.4)

# 使用 OpenCV 的 NMS 进行非极大值抑制
for i, box in zip(keep_indices, boxes_sorted):
    print(f"预测框 {i}, 坐标: {box}")

# 获取原图的尺寸
orig_h, orig_w = original_image.shape[:2]
scale_x, scale_y = orig_w / _dimension[1] , orig_h / _dimension[0]  # 640 是模型输入尺寸

# 创建一张跟原图一样大的图层（或其他颜色）
overlay = np.zeros_like(original_image, dtype=np.uint8)

blended_image = original_image.copy()
cv2.imshow("原图", blended_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

for index in range(len(keep_indices)):
    # 获取 NMS 后的索引 
    #i = indices_nms[index][0]  # OpenCV 返回的是二维数组，所以需要取第一个元素
    i = keep_indices[index]
    
    # 生成随机颜色，范围在 0-255，顺序为 BGR
    random_color = np.random.randint(0, 256, size=(3,), dtype=np.uint8)
    # 应用到 overlay 整张图上
    overlay[:, :] = random_color  # shape (H, W, 3)

    masked_area = get_masks(proto, preds, i)
    # 保存掩膜图像
    #cv2.imwrite(f"masked_area_{i}.png", masked_area)
    masked_area_original = cv2.resize(masked_area, (orig_w, orig_h))
    
    # 创建掩膜的布尔索引（值为 True 的地方要加颜色）
    mask = masked_area_original.astype(bool)  # shape: (H, W
    print("掩膜非零区域数量：", np.count_nonzero(masked_area_original))
   
    # 叠加图层：alpha 混合
    alpha = 0.4  # 半透明程度，0 表示完全透明，1 表示不透明
   
    # 只在掩膜区域混合颜色
    blended_image[mask] = cv2.addWeighted(original_image[mask], 1 - alpha, overlay[mask], alpha, 0)

    # ========== ✅ 新增 OCR 流程 ==========
    # 1. 提取掩膜区域
    bit_region = cv2.bitwise_and(original_image, original_image, mask=masked_area_original)

    # 2. 灰度化并二值化
    #gray = cv2.cvtColor(bit_region, cv2.COLOR_BGR2GRAY)
    #_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #  取掩膜非零区域最小外接矩形
    x, y, w, h = cv2.boundingRect(masked_area_original)
    #cropped_binary = binary[y:y+h, x:x+w]
    #cv2.imshow("Original_mask", bit_region)
    #cv2.imwrite(f"masked_area_{i}.png", cropped_binary)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    # 3. OCR 识别
    #text = pytesseract.image_to_string(binary, config='--psm 11')
    text, ocr_img = get_ocr(bit_region)
    print(f"识别结果 OCR: {'_'.join(text)}")
    #cv2.imshow("Original_mask", ocr_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    draw_multiline_text_with_background(blended_image, text, x, y, h)

# 3. 显示结果
cv2.imshow("叠加图", blended_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



