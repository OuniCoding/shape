import cv2
import os
import numpy as np
import serial
import psutil
import xml.etree.ElementTree as ET
from pathlib import Path

brightness = np.array([0,0,0,0])
def a_data(value):
    brightness[0] = value
def b_data(value):
    brightness[1] = value
def c_data(value):
    brightness[2] = value
def d_data(value):
    brightness[3] = value

process = psutil.Process(os.getpid())
p_core_ids = [0, 1, 2, 3, 4, 5, 6]
# 親和性設置 P-Core
process.cpu_affinity(p_core_ids)
print("Current CPU affinity:", process.cpu_affinity())


cv2.namedWindow('image')
cv2.createTrackbar('CH: A', 'image', 0, 255, a_data)
cv2.createTrackbar('CH: B', 'image', 0, 255, b_data)
cv2.createTrackbar('CH: C', 'image', 0, 255, c_data)
cv2.createTrackbar('CH: D', 'image', 0, 255, d_data)

cv2.setTrackbarPos('CH: A', 'image', brightness[0])
cv2.setTrackbarPos('CH: B', 'image', brightness[1])
cv2.setTrackbarPos('CH: C', 'image', brightness[2])
cv2.setTrackbarPos('CH: D', 'image', brightness[3])

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

