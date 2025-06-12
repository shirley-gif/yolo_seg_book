import cv2
from ultralytics import YOLO

def resize_to_fit_screen(img, max_size=1000):
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def draw_custom_lines(img, keypoints, lines, color=(0,255,255), thickness=10):
    """
    img:      numpy数组图片
    keypoints: [17, 2] 关键点坐标
    lines:    [(a, b), ...] 需要画的线索引（a, b）列表
    color:    线条颜色
    thickness:线条粗细
    """
    for start, end in lines:
        pt1 = tuple(map(int, keypoints[start]))
        pt2 = tuple(map(int, keypoints[end]))
        cv2.line(img, pt1, pt2, color, thickness)
        # 关键点画圆，可选
        cv2.circle(img, pt1, 6, (0, 255, 255), -1)
        cv2.circle(img, pt2, 6, (255, 255, 0), -1)
    return img

img_path = "Weixin.jpg"
model = YOLO("yolov8n-pose.pt")
results = model(img_path)
result = results[0]
# img = result.plot() # 画出原始的骨架
img = cv2.imread(img_path)
kpts = result.keypoints.xy[0]  # [17,2]

'''
0 鼻子 ; 1 左眼 ; 2 右眼; 3 左耳; 4 右耳;5 左肩; 6 右肩;
7 左肘 ; 8 右肘 ; 9 左腕;10 右腕;11 左髋;12右髋;13 左膝;
14右膝 ;15 左踝; 16 右踝
'''

# 定义你想要画的线
custom_lines = [
    
    (9, 10),  # 左手腕-右手腕
    (5, 6),   # 左肩-右肩
    (5,7),  # 左肩-左肘部
    (5,8), # 左肩 - 右肘
    (6,8), #右肩 - 右肘
    (7,9),
    (8,10),
    (9,11),
     (10, 12),
     (11, 13),
     (10,13),
     (11,12),
     (8,12),
     (9,13),
     (13,15),
     (13,14),
     (12,15),
     (12,14),
     (14,16),
     
     

    # 可以加更多自定义线，如(3, 4)左耳-右耳等
]

# 只画自定义线，不画骨架
img = draw_custom_lines(img, kpts, custom_lines, color=(0,255,255), thickness=2)
img = resize_to_fit_screen(img, max_size=1000)
cv2.imshow("Custom Lines", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
