"""
BLOB特征分析(simpleblobdetector使用)
"""

import cv2 as cv
import numpy as np
import os

max_Area = 2000
color_t = 'trans'
work_path = 'temp/'+color_t
if not os.path.exists(work_path):
    os.makedirs(work_path)
if color_t == 'red':
    blur = 15
    thres = 23
elif color_t == 'blue':
    blur = 7
    thres = 23
elif color_t == 'gold':
    blur = 11
    thres = 23
elif color_t == 'green':
    blur = 15
    thres = 20
elif color_t == 'pk':
    blur = 7
    thres = 24
elif color_t == 'lblue':
    blur = 7
    thres = 20
elif color_t == 'white':
    blur = 5
    thres = 88
elif color_t == 'trans':
    blur = 7
    thres = 19 #66
    max_Area = 7000
else:   #black
    blur = 11
    thres = 6

def process_img(frame):
    cv.imshow("input", frame)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #gray = cv.bitwise_not(gray)
    cv.imshow("gray", gray)

    if color_t == 'gold':
        gray2 = cv.medianBlur(gray, blur)
    else:
        gray2 = cv.GaussianBlur(gray, (blur, blur), 0)
    cv.imshow("gray2", gray2)

    # if color_t == 'black' or color_t == 'trans' or color_t == 'gold' or color_t == 'blue' or color_t == 'green' or color_t == 'red':
    if color_t == 'black':
        ret, binary = cv.threshold(gray2, thres, 255, cv.THRESH_BINARY)# + cv.THRESH_OTSU)
        lower = np.array([6, 6, 6])  # 轉換成 NumPy 陣列，範圍稍微變小 ( 55->30, 70->40, 252->200 )
        upper = np.array([27, 27, 25])  # 轉換成 NumPy 陣列，範圍稍微加大 ( 70->90, 80->100, 252->255 )
        output = cv.inRange(frame, lower, upper)   # 取得顏色範圍的顏色
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))  # 設定膨脹與侵蝕的參數
        output = cv.dilate(output, kernel)       # 膨脹影像，消除雜訊
        output9 = cv.erode(output, kernel)        # 縮小影像，還原大小
        cv.imshow("O9", output9)
    else:
        ret, binary = cv.threshold(gray2, thres, 255, cv.THRESH_BINARY)  # + cv.THRESH_OTSU)

    cv.imshow("BIN", binary)
    output5 = cv.adaptiveThreshold(gray2, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 13, 2)
    cv.imshow("O5", output5)
    output7 = cv.subtract(output5, gray2)
    cv.imshow("O7", output7)
    output8 = cv.subtract(output5, output7)
    cv.imshow("O8", output8)

    output7_inv = cv.bitwise_or(binary, gray2) #(output5, output7)
    # if color_t == 'trans':
    #    output7_inv = cv.bitwise_not(output7_inv)

    cv.imshow("O7_inv", output7_inv)

    return binary, output7, output7_inv

def process_blob(binary, output7_inv):
    params = cv.SimpleBlobDetector_Params()

    # change thresholds
    params.minThreshold = 0
    params.maxThreshold = 120

    # filter by color
    params.filterByColor = True
    params.blobColor = 0

    # filter by area
    params.filterByArea = True
    params.minArea = 85
    params.maxArea = max_Area   #2000   trans:7000

    # filter by circularity
    params.filterByCircularity = False  #True
    params.minCircularity = 0.01
    params.maxCircularity = 150

    # Filter by Convexity
    params.filterByConvexity = False  #True
    params.minConvexity = 1

    # Filter by Inertia
    params.filterByInertia = False  #True
    params.minInertiaRatio = 0.5

    # contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # contours, hierarchy = cv.findContours(output9, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in range(len(contours)):
        area = cv.contourArea(contours[cnt])
        if area >= 85 and area <= max_Area:     # trans:7000
            # 提取與繪制輪廓
            cv.drawContours(frame, contours, cnt, (0, 255, 0), 1)
            print(cnt,':', area)
        else:
            cv.drawContours(frame, contours, cnt, (0, 25, 255), 1)
    cv.imshow("contours", frame)
    # 提取关键点
    detector = cv.SimpleBlobDetector_create(params)
    keypoints = detector.detect(output7_inv)    # gray
    print(len(keypoints))
    result = []
    if len(keypoints) > 0:
        for marker in keypoints:
            cv.circle(frame, (int(marker.pt[0]), int(marker.pt[1])), 3, (255, 0, 255), -1, 8)
            # cv.circle(frame, (int(marker.pt[0]), int(marker.pt[1])), int(marker.size/2), (0, 0, 255), 2, 1)
            #print(':', marker.size)

            result = cv.drawMarker(frame, tuple(int(i) for i in marker.pt), color=(0, 255, 0))
        cv.imshow("result", result)
        # cv.imwrite('temp/trans/result_' + color_t + '.png', result)
    return result

img_path = 'F:\\project\\bottlecap\\test\\' + color_t + "\\"
img_files = os.listdir(img_path)
for img_file in img_files:
    frame = cv.imread(img_path+img_file)
    img_bin, img_o7, img_o7_inv = process_img(frame)
    cv.imwrite('temp/'+color_t+ '_p/'+ img_file + '_o7.png', img_o7)
    cv.imwrite('temp/'+color_t+ '_p/'+ img_file + '_o7_inv.png', img_o7_inv)
    img_res = process_blob(img_bin, img_o7_inv)
    if len(img_res) > 0:
        cv.imwrite('temp/'+color_t+'/result_' + img_file + '.png', img_res)
    cv.waitKey(1)


cv.waitKey(0)
cv.destroyAllWindows()