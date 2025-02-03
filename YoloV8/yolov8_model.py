from ultralytics import YOLO

# 載入不同的模型
model_n = YOLO('./model/yolov8n.pt')   # Nano
model_s = YOLO('./model/yolov8s.pt')   # Small
model_m = YOLO('./model/yolov8m.pt')   # Medium
model_l = YOLO('./model/yolov8l.pt')   # Large
model_x = YOLO('./model/yolov8x.pt')   # Xtreme