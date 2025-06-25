import cv2
import numpy as np
import onnxruntime as ort
from yolo11m_onnx_book2call import (
    predict, get_masks,
    filter_and_sort_boxes, get_ocr, draw_multiline_text_with_background
)


def preprocess_image_from_frame(frame, input_size=(640, 640)):
    image = cv2.resize(frame, input_size)
    img_rgb = image[:, :, ::-1]
    img_tensor = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_tensor = np.expand_dims(img_tensor, axis=0)
    return img_tensor, frame.copy(), None

_threshold = 0.85
_dimension = (640, 640)
session = ort.InferenceSession("runs/segment/bookspine-seg-resume/weights/best.onnx", providers=["CPUExecutionProvider"])

cap = cv2.VideoCapture('dku_data/video_bookshelf.mp4')  # Use 0 for webcam or provide video path

# 获取视频的帧宽、高和帧率
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 创建 VideoWriter 对象，保存为 MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 编码格式
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (frame_width, frame_height))


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img, original_image, _ = preprocess_image_from_frame(frame, _dimension)
    outputs = predict(session, img)
    proto = outputs[1][0]
    preds = outputs[0][0]

    keep_indices, boxes_sorted = filter_and_sort_boxes(preds, _threshold, 0.4)
    orig_h, orig_w = original_image.shape[:2]
    overlay = np.zeros_like(original_image, dtype=np.uint8)
    blended_image = original_image.copy()

    for index in range(len(keep_indices)):
        i = keep_indices[index]
        random_color = np.random.randint(0, 256, size=(3,), dtype=np.uint8)
        overlay[:, :] = random_color

        masked_area = get_masks(proto, preds, i)
        masked_area_original = cv2.resize(masked_area, (orig_w, orig_h))
        if masked_area_original is None or np.count_nonzero(masked_area_original) == 0:
            continue  # 跳过无效掩膜区域

        mask = masked_area_original.astype(bool)
        alpha = 0.4
        blended_image[mask] = cv2.addWeighted(original_image[mask], 1 - alpha, overlay[mask], alpha, 0)

        bit_region = cv2.bitwise_and(original_image, original_image, mask=masked_area_original)
        x, y, w, h = cv2.boundingRect(masked_area_original)

        #text, ocr_img = get_ocr(bit_region)
        #draw_multiline_text_with_background(blended_image, text, x, y, h)

    out.write(blended_image)
    cv2.imshow("Video OCR", blended_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
