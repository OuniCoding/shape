# 🚀 YOLOv3 轉 YOLOv8 的完整流程
## 步驟 1：環境準備
1. 安裝 YOLOv8（Ultralytics）

```commandline
 pip install ultralytics
```
2. 確認依賴

    +  Python ≥ 3.8

    + PyTorch ≥ 1.8

    + OpenCV、NumPy 等常見庫

***
## 步驟 2：資料集格式轉換

YOLOv3 和 YOLOv8 都使用 YOLO 格式，但 YOLOv8 的格式更簡潔，並且支援分割（Segmentation）等任務。

✅ YOLOv3 格式範例：
```text
    0 0.5 0.5 0.4 0.3
```
+ 0 → 類別 ID

+ 0.5 0.5 → 邊界框中心座標（歸一化）

+ 0.4 0.3 → 邊界框寬度和高度（歸一化）

### 🔄 YOLOv8 格式變化：
+ 基本的物件檢測與 YOLOv3 類似。

+ 若使用分割任務，則會包含多邊形點座標：
    
```text
    0 0.5 0.5 0.4 0.3 0.1 0.2 0.3 0.4 ...
```

### 轉換工具：

使用 Python 轉換標註格式：
```python
import os

def convert_yolov3_to_yolov8(source_folder, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for filename in os.listdir(source_folder):
        if filename.endswith('.txt'):
            with open(os.path.join(source_folder, filename), 'r') as file:
                lines = file.readlines()

            with open(os.path.join(target_folder, filename), 'w') as file:
                for line in lines:
                    parts = line.strip().split()
                    class_id = parts[0]
                    bbox = parts[1:5]  # (x_center, y_center, width, height)
                    # YOLOv8 標註不需要改動物件檢測部分
                    file.write(f"{class_id} {' '.join(bbox)}\n")

convert_yolov3_to_yolov8('yolov3_labels', 'yolov8_labels')
```
***

## 步驟 3：建立 YOLOv8 的資料集配置檔
建立一個 data.yaml 文件：
```yaml
train: /path/to/train/images
val: /path/to/val/images

nc: 3  # 類別數量
names: ['class1', 'class2', 'class3']  # 類別名稱
```
***

## 步驟 4：選擇 YOLOv8 模型架構
YOLOv8 提供不同大小的模型：

+ YOLOv8n（Nano） - 最輕量化
+ YOLOv8s（Small） - 適合嵌入式設備
+ YOLOv8m（Medium） - 性能與速度平衡
+ YOLOv8l（Large） - 高精度模型
+ YOLOv8x（Extra Large） - 最高性能

### 🚀 YOLOv8 官方模型下載連結

|模型版本|下載連結|大小| 適用場景 |
|------|-------|----|-----|
|YOLOv8n (Nano)|[下載 YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt)|~6 MB|超輕量級，適合嵌入式裝置|
YOLOv8s (Small)|[下載 YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt)|~20 MB|快速推論，適合中等設備|
YOLOv8m (Medium)|[下載 YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt)|~50 MB|性能與速度平衡|
YOLOv8l (Large)|[下載 YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt)|~80 MB|高精度檢測|
YOLOv8x (Xtreme)|[下載 YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt)|~130 MB|最佳精度，適合高端硬體|

### ⚡ 使用 Python 自動下載模型

+ 自動下載模型：
```python
from ultralytics import YOLO

# 載入不同的模型
odel_n = YOLO('./model/yolov8n.pt')   # Nano
model_s = YOLO('./model/yolov8s.pt')   # Small
model_m = YOLO('./model/yolov8m.pt')   # Medium
model_l = YOLO('./model/yolov8l.pt')   # Large
model_x = YOLO('./model/yolov8x.pt')   # Xtreme
```
執行後，模型會自動下載至 ~/model/ 目錄。

### 🎯 其他模型格式支援
你也可以將 YOLOv8 模型轉換成其他格式，如 ONNX、TensorRT：
```commandline
yolo export model=yolov8s.pt format=onnx
yolo export model=yolov8s.pt format=engine  # TensorRT
```

***

## 步驟 5：重新訓練模型
使用 YOLOv8 進行訓練：
```commandline
yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=100 imgsz=640 batch=16
```
+ task=detect → 目標檢測任務
+ model=yolov8s.pt → 選擇模型
+ data=data.yaml → 指定資料集
+ epochs=100 → 訓練輪數
+ imgsz=640 → 圖像尺寸
+ batch=16 → 批次大小
***

## 步驟 6：模型評估與測試
測試模型效果：

```commandline
yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=data.yaml
```
進行推論：

```commandline
yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source=path_to_images
```

***

## 步驟 7：模型匯出（可選）
將模型導出為 ONNX、TensorRT 等格式：

```commandline
yolo export model=runs/detect/train/weights/best.pt format=onnx
```
***

## ⚡ 優化建議
1. 數據增強：

    + YOLOv8 支援 Mosaic、MixUp、HSV 變化等先進數據增強技術。

2. 超參數調整：

    + 嘗試不同的學習率和批次大小，以獲得更好的結果。

3. FP16 精度訓練：

    + 如果使用 RTX 4070 Ti 或 4060 Ti，可以加速訓練：
```commandline
yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=100 imgsz=640 batch=16 device=0 --half
```

***

## 🎯 常見問題
+ 標註轉換錯誤？  

  檢查標註座標是否仍保持歸一化（0~1 之間）。

+ 模型表現下降？

  嘗試進行超參數微調，並使用更大或更適合的模型架構（如 YOLOv8m 或 YOLOv8l）。

+ 推論速度慢？

  可以使用 TensorRT 或 ONNX 進行模型加速。

***
