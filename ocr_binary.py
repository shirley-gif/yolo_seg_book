import easyocr
import cv2

reader = easyocr.Reader([ 'en'], gpu=False)

img = cv2.imread('masked_area_8190.png')  # 确保图像路径正确
results = reader.readtext(img)

for box, text, conf in results:
    print(f"OCR: {text} （置信度: {conf:.2f}）")



from paddleocr import PaddleOCR
# 初始化 PaddleOCR 实例
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

# 对示例图像执行 OCR 推理 
result = ocr.predict(
    input="masked_area_8190.png")
    
# 可视化结果并保存 json 结果
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")