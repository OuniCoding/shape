import cv2
import os
import numpy as np
import psutil
import xml.etree.ElementTree as ET
from pathlib import Path

def set_blob_param(category,para_name):
    param_file = ET.parse(para_name)
    root = param_file.getroot()
    id = 0
    param = []
    while root[0][id].tag != category:
        id += 1
    for i in range(0, 16):
        param.append(int(root[0][id][i].attrib['value']))

    return param

def read_path_color(param_file):

    ini_file = open(param_file, 'r')
    color = ini_file.readline()
    color = color.replace('\n','')
    path = ini_file.readline()
    path = path.replace('\n','')

    ini_file.close()

    return color, path


def manual_white_balance(img):
    # 將圖像轉換為浮點型，進行計算
    result = img.astype(np.float32)

    # 計算每個通道的均值
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])

    print(avg_r, avg_g, avg_b)

    # 計算白平衡增益
    k_b = avg_g / avg_b * 1.05
    k_r = avg_g / avg_r * 0.9

    # 調整每個通道，達到白平衡效果
    result[:, :, 0] = result[:, :, 0] * k_b  # 調整藍色通道
    result[:, :, 2] = result[:, :, 2] * k_r  # 調整紅色通道

    # 將結果轉換回 8 位整數，並剪切到 [0, 255] 範圍
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def balanecd_process(img):
    # 創建 SimpleWB 物件
    wb = cv2.xphoto.createSimpleWB()

    # 設置白平衡增益和色溫
    wb.setP(0.5)  # p 用於設置簡單白平衡的參數，0.5 為默認值，可以手動調整

    # 應用白平衡
    balanced_img = wb.balanceWhite(img)

    # 顯示處理後的影像
    cv2.imshow('Balanced Image', balanced_img)

    # 手動白平衡處理
    m_balanced_img = manual_white_balance(img)

    # 顯示處理後的影像
    cv2.imshow('Manual Balanced Image', m_balanced_img)

    return
#image = cv2.imread('F:\\project\\bottlecap\\test1\\Image_20240621113923008.jpg')
#img_path = 'f:\\project\\bottlecap\\test1\\out\\green\\2024-09-19\\2\\resultNG\\'
# img_path = 'e:\\logo\\red\\2024-10-30\\1\\resultG\\' #'F:\\Temp\\report\\report\\image\\'#
# img_path = 'F:\\project\\玻璃瓶\\sample\\'

param_file = 'param.ini'
color, img_path = read_path_color(param_file)
bright = 1
img_files = [_ for _ in os.listdir(img_path) if (_.endswith('.jpg') or _.endswith('.bmp') or _.endswith('.png'))]
img_file = img_files[0]
#image = cv2.imread('D:\\project\\bottlecap\\test1\\in\\green\\2024-07-10\\1\\resultG\\20240710_14-31-31_029.jpg')
#image = cv2.imread(img_path+img_file)

# -----------------------------------------------------------------------------------
# 使用 numpy 讀取文件
with open(img_path+img_file, 'rb') as f:
    image_data = f.read()

# 將讀取到的數據轉換為 numpy 數組
image_array = np.frombuffer(image_data, np.uint8)

# 使用 cv2.imdecode 解碼圖像解
image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
image = cv2.addWeighted(image, bright, image, 0, 0)  # 0517 bright
w = image.shape[1]
h = image.shape[0]
image_s = cv2.resize(image, (int(w / 2), int(h / 2)))
# -----------------------------------------------------------------------------------

cv2.imshow('BGR', image_s)
# balanecd_process(image_s)
if param_file == 'param_out.ini':
    param = set_blob_param(color, 'Settings/oblob-param-newcam.xml')
else:
    param = set_blob_param(color, 'Settings/iblob-param-newcam.xml')
#white in
#hsv_low = np.array([0, 0, 98])
#hsv_high = np.array([221, 38, 255])
#white out
#hsv_low = np.array([0, 0, 54])
#hsv_high = np.array([221, 31, 220])

hsv_low = np.array([param[6], param[8], param[10]])
hsv_high = np.array([param[7], param[9], param[11]])

hsv_low[0] = param[1]

def h_low(value):
    hsv_low[0] = value
def h_high(value):
    hsv_high[0] = value
def s_low(value):
    hsv_low[1] = value
def s_high(value):
    hsv_high[1] = value
def v_low(value):
    hsv_low[2] = value
def v_high(value):
    hsv_high[2] = value

process = psutil.Process(os.getpid())
p_core_ids = [0, 1, 2, 3, 4, 5, 6]
# 親和性設置 P-Core
process.cpu_affinity(p_core_ids)
print("Current CPU affinity:", process.cpu_affinity())


cv2.namedWindow('image')
cv2.createTrackbar('bin low', 'image', 0, 255, h_low)
#cv2.createTrackbar('H high', 'image', 0, 255, h_high)
#cv2.createTrackbar('S low', 'image', 0, 255, s_low)
#cv2.createTrackbar('S high', 'image', 0, 255, s_high)
#cv2.createTrackbar('V low', 'image', 0, 255, v_low)
#cv2.createTrackbar('V high', 'image', 0, 255, v_high)

cv2.setTrackbarPos('bin low', 'image', hsv_low[0])
#cv2.setTrackbarPos('S low', 'image', hsv_low[1])
#cv2.setTrackbarPos('V low', 'image', hsv_low[2])
#cv2.setTrackbarPos('H high', 'image', hsv_high[0])
#cv2.setTrackbarPos('S high', 'image', hsv_high[1])
#cv2.setTrackbarPos('V high', 'image', hsv_high[2])

index=0
while True:
    #dst = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #dst = cv2.inRange(dst, hsv_low, hsv_high)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray, (param[0], param[0]), 0)
    ret, dst = cv2.threshold(gray2, hsv_low[0], 255, cv2.THRESH_BINARY)
    cv2.imshow('dst', dst)
    cv2.imshow('zoom', cv2.resize(dst, (560, 560))) #(560, 560)
    key = cv2.waitKeyEx(1)
    if key & 0xff == ord('q') or key & 0xff == ord('Q'):
        break
    elif key == 2490368 or key == 2424832:
        index -= 1;
        if index < 0:
            index = len(img_files)-1
        img_file = img_files[index]
        # image = cv2.imread('D:\\project\\bottlecap\\test1\\in\\green\\2024-07-10\\1\\resultG\\20240710_14-31-31_029.jpg')
        # image = cv2.imread(img_path+img_file)
        # -----------------------------------------------------------------------------------
        # 使用 numpy 讀取文件
        with open(img_path + img_file, 'rb') as f:
            image_data = f.read()

        # 將讀取到的數據轉換為 numpy 數組
        image_array = np.frombuffer(image_data, np.uint8)

        # 使用 cv2.imdecode 解碼圖像解
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        w = image.shape[1]
        h = image.shape[0]
        image_s = cv2.resize(image, (int(w / 2), int(h / 2)))
        # image = cv2.resize(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)))
        # -----------------------------------------------------------------------------------

        cv2.imshow('BGR', image_s)
        print(img_file)
        # balanecd_process(image_s)
    elif key == 2621440 or key == 2555904:
        index += 1;
        if index >= len(img_files):
            index = 0
        img_file = img_files[index]
        # image = cv2.imread('D:\\project\\bottlecap\\test1\\in\\green\\2024-07-10\\1\\resultG\\20240710_14-31-31_029.jpg')
        #image = cv2.imread(img_path+img_file)
        # -----------------------------------------------------------------------------------
        # 使用 numpy 讀取文件
        with open(img_path + img_file, 'rb') as f:
            image_data = f.read()

        # 將讀取到的數據轉換為 numpy 數組
        image_array = np.frombuffer(image_data, np.uint8)

        # 使用 cv2.imdecode 解碼圖像解
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        w = image.shape[1]
        h = image.shape[0]
        image_s = cv2.resize(image, (int(w / 2), int(h / 2)))
        # image = cv2.resize(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)))
        # -----------------------------------------------------------------------------------

        cv2.imshow('BGR', image_s)
        print(img_file)
        # balanecd_process(image_s)

cv2.destroyAllWindows()

