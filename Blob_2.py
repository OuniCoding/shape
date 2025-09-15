"""
BLOB特征分析(simpleblobdetector使用)
"""

import cv2 as cv
import numpy as np
import os
import time

max_Area = 5000
color_t = 'black'
work_path = 'temp/'+color_t
minSize = 85
blockSize = 13
C_V = 2
if not os.path.exists(work_path):
    os.makedirs(work_path)
    os.makedirs(work_path + '_p/')

if color_t == 'red':
    blur = 15
    thres = 50  #23
    minSize = 50
elif color_t == 'blue':
    blur = 7
    thres = 25 #23
elif color_t == 'gold':
    blur = 11
    thres = 65  #23
elif color_t == 'green':
    blur = 15
    thres = 44
elif color_t == 'pink':
    blur = 7
    thres = 64
elif color_t == 'lightblue':
    blur = 15
    thres = 70  #20
elif color_t == 'white':
    blur = 7    #5
    thres = 69 #112 #88
    # color_t = color_t + 'NG'
    max_Area = 5000
elif color_t == 'trans':
    blur = 7
    thres = 19 #66
    max_Area = 7000
else:   #black
    blur = 11
    thres = 100 #240
    # color_t = color_t + '1'
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
        lower = np.array([0, 0, 0])  # [6, 6, 6]轉換成 NumPy 陣列，範圍稍微變小 ( 55->30, 70->40, 252->200 )
        upper = np.array([48, 255, 15])  # [27, 27, 25]轉換成 NumPy 陣列，範圍稍微加大 ( 70->90, 80->100, 252->255 )
        output = cv.inRange(frame, lower, upper)   # 取得顏色範圍的顏色
        #kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))  # 設定膨脹與侵蝕的參數
        #output = cv.dilate(output, kernel)       # 膨脹影像，消除雜訊
        #output = cv.erode(output, kernel)        # 縮小影像，還原大小
        output9 = cv.bitwise_and(gray2, gray2, mask=output)
        ret, output9 = cv.threshold(output9, 20, 255, cv.THRESH_BINARY)
        cv.imshow("O", output)
        cv.imshow("O9", output9)
        cv.imshow("BIN", binary)
        output5 = cv.adaptiveThreshold(binary, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize, C_V)
        #output5 = cv.adaptiveThreshold(output, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize, C_V)
        cv.imshow("O5", output5)
        output7 = cv.subtract(output5, gray2)
        cv.imshow("O7", output7)
        output8 = cv.subtract(output5, output7)
        cv.imshow("O8", output8)

        # output7_inv = cv.bitwise_or(binary, gray2) #(output5, output7)
        output7_inv = cv.subtract(binary, gray2) #(output5, output7)
        # if color_t == 'trans':
        #    output7_inv = cv.bitwise_not(output7_inv)

        cv.imshow("O7_inv", output7_inv)

        return binary, output7, output #output7_inv

    else:
        ret, binary = cv.threshold(gray2, thres, 255, cv.THRESH_BINARY)  # + cv.THRESH_OTSU)

    cv.imshow("BIN", binary)
    output5 = cv.adaptiveThreshold(gray2, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize, C_V)
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

def process_blob(file, binary, output7_inv):
    params = cv.SimpleBlobDetector_Params()

    # change thresholds
    params.minThreshold = 0
    params.maxThreshold = 120   #125   #120

    # 重复的最小次数，只有属于灰度图像斑点的那些二值图像斑点数量大于该值时，该灰度图像斑点才被认为是特征点
    params.minRepeatability = 2
    # 最小的斑点距离，不同二值图像的斑点间距离小于该值时，被认为是同一个位置的斑点，否则是不同位置上的斑点
    params.minDistBetweenBlobs = 10

    # filter by color
    params.filterByColor = True
    params.blobColor = 0

    # filter by area
    params.filterByArea = True
    params.minArea = minSize    #50 #85
    params.maxArea = max_Area   #2000   trans:7000

    # filter by circularity
    params.filterByCircularity = False  #True
    params.minCircularity = 0.01
    params.maxCircularity = 150

    # Filter by Convexity
    params.filterByConvexity = False  #True #
    params.minConvexity = 0.1

    # Filter by Inertia
    params.filterByInertia = False  #True   #False  #
    params.minInertiaRatio = 0.1    #0.5

    # contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # contours, hierarchy = cv.findContours(output9, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#    for cnt in range(len(contours)):
#        area = cv.contourArea(contours[cnt])
#        if area >= minSize and area <= max_Area:     # trans:7000
#            # 提取與繪制輪廓
#            cv.drawContours(frame, contours, cnt, (0, 255, 0), 1)
#            print(cnt,':', area)
#        else:
#            cv.drawContours(frame, contours, cnt, (0, 25, 255), 1)
#    #w cv.imshow("contours", frame)
    # 提取关键点
    detector = cv.SimpleBlobDetector_create(params)
    keypoints = detector.detect(output7_inv)    # (binary) # gray
    print(len(keypoints))
    result = []
    blank = np.zeros((1, 1))
    if len(keypoints) > 0:
        result = cv.drawKeypoints(
            frame, keypoints, blank, (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        #for marker in keypoints:
        #    cv.circle(frame, (int(marker.pt[0]), int(marker.pt[1])), 3, (255, 0, 255), -1, 8)
        #    # cv.circle(frame, (int(marker.pt[0]), int(marker.pt[1])), int(marker.size/2), (0, 0, 255), 2, 1)
        #    print('size:', marker.size)
        #    result = cv.drawMarker(frame, tuple(int(i) for i in marker.pt), color=(0, 255, 0))
        cv.putText(result, str(len(keypoints)), (5, 20), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1)
        cv.imshow("result", result)
        cv.imwrite('temp/'+ color_t +' /result_' + file, result)
    return result

img_path = 'F:\\project\\bottlecap\\Samples\\' + color_t + "\\"
# img_files = os.listdir(img_path)
img_files = [_ for _ in os.listdir(img_path) if (_.endswith('.jpg') or _.endswith('.png'))]
for img_file in img_files:
    prev_time = time.time()
    print(img_file)
    frame = cv.imread(img_path+img_file)
    img_bin, img_o7, img_o7_inv = process_img(frame)
    cv.imwrite('temp/' + color_t + '_p/' + img_file + '_b.png', img_bin)
    cv.imwrite('temp/'+color_t+ '_p/'+ img_file + '_o7.png', img_o7)
    cv.imwrite('temp/'+color_t+ '_p/'+ img_file + '_o7_inv.png', img_o7_inv)
    img_res = process_blob(img_file, img_bin, img_o7_inv)
    if len(img_res) > 0:
        cv.imwrite('temp/'+color_t+'/result_' + img_file + '.png', img_res)
    cv.waitKey(1)
    now_time = time.time()
    fps = (1 / (now_time - prev_time))
    print("FPS: {:.3f}".format(fps))

cv.waitKey(0)
cv.destroyAllWindows()