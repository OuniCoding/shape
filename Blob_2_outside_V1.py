"""
BLOB特征分析(simpleblobdetector使用)
"""

import cv2 as cv
import numpy as np
import os
import time


def process_img(frame):
    cv.imshow("input", frame)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #gray = cv.bitwise_not(gray)
    cv.imshow("gray", gray)

    if color_t == 'gold':
        #gray2 = cv.medianBlur(gray, blur)
        gray2 = cv.GaussianBlur(gray, (blur, blur), 0)
    else:
        gray2 = cv.GaussianBlur(gray, (blur, blur), 0)
    cv.imshow("gray2", gray2)

    # if color_t == 'black' or color_t == 'trans' or color_t == 'gold' or color_t == 'blue' or color_t == 'green' or color_t == 'red':
    if color_t == 'gold':
        ret, binary = cv.threshold(gray2, thres, 255, cv.THRESH_BINARY)# + cv.THRESH_OTSU)
        #lower = np.array([6, 6, 6])  # 轉換成 NumPy 陣列，範圍稍微變小 ( 55->30, 70->40, 252->200 )
        #upper = np.array([27, 27, 25])  # 轉換成 NumPy 陣列，範圍稍微加大 ( 70->90, 80->100, 252->255 )
        lower = np.array([0, 0, 55])  # 轉換成 NumPy 陣列，範圍稍微變小 ( 55->30, 70->40, 252->200 )
        upper = np.array([36, 255, 98])  # 轉換成 NumPy 陣列，範圍稍微加大 ( 70->90, 80->100, 252->255 )
        output = cv.inRange(frame, lower, upper)   # 取得顏色範圍的顏色
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (blur, blur))  # 設定膨脹與侵蝕的參數
        output = cv.dilate(output, kernel)       # 膨脹影像，消除雜訊
        #output9 = cv.erode(output, kernel)  # 縮小影像，還原大小
        output = cv.erode(output, kernel)        # 縮小影像，還原大小
        output9 = cv.bitwise_and(gray, gray2, mask=output)
        ret, output9 = cv.threshold(output9, 10, 255, cv.THRESH_BINARY_INV)
        cv.imshow("O9", output9)

        cv.imshow("BIN", binary)
        output5 = cv.adaptiveThreshold(binary, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize, C_V)
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

        # return binary, output7, output9 #output7_inv

    else:
        ret, binary = cv.threshold(gray2, thres, 255, cv.THRESH_BINARY)  # + cv.THRESH_OTSU)

    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    black = cv.imread('black1024.jpg')
    amount = 0
    """    
    # 最大內接圓
    raw_dist = np.empty(binary.shape, dtype=float)
    for cnt in range(len(contours)):
        (x, y), radius = cv.minEnclosingCircle(contours[cnt])
        center = (int(x), int(y))
        radius = int(radius)
        if radius < 500:
            continue
        for i in range(binary.shape[0]):
            for j in range(binary.shape[1]):
                raw_dist[i, j] = cv.pointPolygonTest(contours[cnt], (i, j), True)
        minVal, maxVal, _, maxDistpt = cv.minMaxLoc(raw_dist)
        minVal = abs(minVal)
        maxVal = abs(maxVal)
        radius = int(maxVal)
        cv.circle(black, maxDistpt, radius, (255, 255, 255), -1)
        print(radius)
    """
    # 最小外接圓
    for cnt in range(len(contours)):
        (x, y), radius = cv.minEnclosingCircle(contours[cnt])
        center = (int(x), int(y))
        radius = int(radius)
        # debug
        #print('r=',radius)
        # debug

        if radius < 450:
            continue
        if radius > 480:
            radius = 462
        # debug
        print('r=',radius)
        # debug
        # cv.circle(black, center, radius,(255, 255, 255), -1)
        # 提取與繪制輪廓
        # cv.drawContours(frame, contours, cnt, (0, 255, 0), -1)
        # 輪廓逼近
        epsilon = 0.001 * cv.arcLength(contours[cnt], True)
        approx = cv.approxPolyDP(contours[cnt], epsilon, True)

        # print("approx=",approx)     # 輪廓邊線座標

        # 求解中心位置
        mm = cv.moments(contours[cnt])
        # print(corners,shape_type, mm)
        try:
            if True:
                cx = int(mm['m10'] / mm['m00'])
                cy = int(mm['m01'] / mm['m00'])
        except:
            continue

        # 分析幾何形狀
        corners = len(approx)
        # looking bottom coordinate
        y_max = 0
        x_max = 0
        y_min = 1024
        x_min = 1024
        y = 0
        x = 0
        dis_max = 0
        for c in range(corners):
            [xy] = approx[c]
            y = xy[1:2]
            x = xy[:1]
            # cv.circle(black, (int(x), int(y)), 5, (0, 255, 0), -1)
            # debug
            # cv.line(black, (int(x), int(y)), (cx, cy), (0, 0, 255), 1)
            # debug
            if y > y_max:  # x > x_max :  #
                y_max = int(y)
                x_max = int(x)
                [xy_previous] = approx[c - 1]
            if y < y_min:
                y_min = int(y)
                x_min = int(x)

        #dis = int((((x_max - center[0]) ** 2 + (y_max - center[1]) ** 2) ** 0.5))
        dis = int((((x_max - x_min) ** 2 + (y_max - y_min) ** 2) ** 0.5) / 2 * 1)
        print(x_max, y_max, x_min, y_min, dis, center)
        d_x = abs((x_max - x_min) / 2)
        d_y = abs((y_max - y_min) / 2)
        if x_max > x_min:
            cx1 = int(x_min + d_x)
        else:
            cx1 = int(x_max + d_x)
        cy1 = int(y_min + d_y)
        print('cx1=', cx1, ' cy1=', cy1)
        print('cx=', cx, 'cy=', cy)
        if (x_max >= cx and x_min >= cx) or (x_max <= cx and x_min <= cx):
            cx1 = cx
        #if radius > dis: radius = dis

        cv.circle(black, (cx, cy), radius, (255, 255, 255), -1)
        # cv.circle(black, (cx, cy), dis, (255, 255, 255), -1)
        # cv.circle(black, center, dis, (0, 255, 0), 2)
        # cv.circle(black, (cx1, cy1), radius, (255, 255, 255), -1)
        cv.circle(black, (cx, cy), 5, (0, 255, 0), -1)
        cv.circle(black, (cx1, cy1), 2, (0, 0, 255), -1)
        # debug
        # cv.circle(black, (cx1, cy1), dis, (0, 0, 255), 1)
        # cv.imshow('line', black)
        # cv.waitKey(0)
        # debug

        # 求解中心位置
        # mm = cv.moments(contours[cnt])
        # # print(corners,shape_type, mm)
        # try:
        #     if True:
        #         cx = int(mm['m10'] / mm['m00'])
        #         cy = int(mm['m01'] / mm['m00'])
        #         cv.circle(black, (cx, cy), 2, (0, 0, 255), -1)
        #
        #         amount = amount + 1
        #         # 顏色分析
        #         #color = frame[cy][cx]
        #         #color_str = "(" + str(color[0]) + ", " + str(color[1]) + ", " + str(color[2]) + ")"
        #         #cv.putText(result, str(amount), (cx+5, cy+5), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1)
        #
        #         # 計算面積與周長
        #         # p = cv.arcLength(contours[cnt], True)
        #         # area = cv.contourArea(contours[cnt])
        #
        #         # y = xy_previous[1]
        #         # x = xy_previous[0]
        #         #dis = ((x-x_max)**2+(y-y_max)**2)**0.5
        #         #print(xy_previous)
        #
        #         dis = int(((cx - x_max) ** 2 + (cy - y_max) ** 2) ** 0.5)
        #         print(x_max, y_max, cx, cy, dis)
        #         print(cx, cy)
        #
        #         #print("%s 中心座標: %s 周長: %.3f, 面積: %.3f 顏色: %s 形狀: %s (%2i) 底部座標(%3i, %3i)" % (amount, (cx,cy), p, area, color_str, shape_type, corners, x_max, y_max))
        #         #print("%s 中心座標: %s 顏色: %s 形狀: %s (%2i) 底部座標(%3i, %3i) 距離: %3.3f" % (amount, (cx, cy), color_str, shape_type, corners, x_max, y_max, dis))
        #
        #         # mark bottom coordinate
        #         #if radius < dis_max:
        #         #    radius = int(dis_max) #y_max - cy
        #         #cv.circle(black, (int(cx - abs(radius-dis_max)), cy), radius, (255, 255, 255), -1)
        #         cv.circle(black, (cx, cy), radius, (255, 255, 255), -1)
        #         cv.circle(black, (cx, cy), 2, (0, 0, 255), -1)
        # except:
        #     continue

    cv.imshow("f", black)
    black_b = cv.cvtColor(black, cv.COLOR_BGR2GRAY)
    ret, black_b = cv.threshold(black_b, 1, 255, cv.THRESH_BINARY)
    cv.imshow("b", black_b)
    binary = cv.bitwise_and(binary, black_b)

    cv.imshow("BIN", binary)
    output5 = cv.adaptiveThreshold(gray2, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize, C_V)
    cv.imshow("O5", output5)
    output7 = cv.subtract(output5, gray2)
    cv.imshow("O7", output7)
    output8 = cv.subtract(output5, output7)
    cv.imshow("O8", output8)

    output7_inv = cv.bitwise_and(binary, output5) #(output5, output7)
    #output7_inv = cv.bitwise_or(binary, output5)
    # if color_t == 'trans':
    #    output7_inv = cv.bitwise_not(output7_inv)
    output7_inv = cv.bitwise_and(output7_inv, black_b)

    cv.imshow("O7_inv", output7_inv)
    # debug
    # cv.waitKey(0)
    # debug
    return binary, output7, output7_inv

def process_blob(file,frame, binary, output7_inv):
    params = cv.SimpleBlobDetector_Params()

    # change thresholds
    params.minThreshold = 0
    params.maxThreshold = 200   #120

    # 重复的最小次数，只有属于灰度图像斑点的那些二值图像斑点数量大于该值时，该灰度图像斑点才被认为是特征点
    params.minRepeatability = 2
    # 最小的斑点距离，不同二值图像的斑点间距离小于该值时，被认为是同一个位置的斑点，否则是不同位置上的斑点
    params.minDistBetweenBlobs = 10

    # filter by color
    params.filterByColor = False    #True
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
    params.filterByConvexity = False  #True
    params.minConvexity = 0.1

    # Filter by Inertia
    params.filterByInertia = False  #True
    params.minInertiaRatio = 0.1    #0.5

    # contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # contours, hierarchy = cv.findContours(output9, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in range(len(contours)):
        area = cv.contourArea(contours[cnt])
        print(cnt, ':', area)
        if area >= minSize and area <= max_Area:     # trans:7000
            # 提取與繪制輪廓
            cv.drawContours(frame, contours, cnt, (0, 255, 0), 1)
            #print(cnt,':', area)
        else:
            cv.drawContours(frame, contours, cnt, (0, 25, 255), 1)
    cv.imshow("contours", frame)
    # 提取关键点
    keypoints = []
    detector = cv.SimpleBlobDetector_create(params)
    if color_t != 'black':
        keypoints = detector.detect(binary) #(output7_inv)    #  gray
    else:
        keypoints = detector.detect(binary) #(output7_inv)  # (binary) #  gray
    print(len(keypoints))
    result = []
    blank = np.zeros((1, 1))
    if len(keypoints) > 0:
        result = cv.drawKeypoints(
            frame, keypoints, blank, (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

#        for marker in keypoints:
#            cv.circle(frame, (int(marker.pt[0]), int(marker.pt[1])), 3, (255, 0, 255), -1, 8)
#            # cv.circle(frame, (int(marker.pt[0]), int(marker.pt[1])), int(marker.size/2), (0, 0, 255), 2, 1)
#            print('size:', marker.size)
#
#            result = cv.drawMarker(frame, tuple(int(i) for i in marker.pt), color=(0, 255, 0))
        cv.putText(result, str(len(keypoints)), (5, 20), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1)
        cv.imshow("result", result)
        cv.imwrite('temp_o/'+ color_t +' /result_' + file, result)
    return result

max_Area = 5000
color_t = 'white'
work_path = 'temp_o/'+color_t
minSize = 100
sens = 7
rate = (11 - sens) / 5

blockSize = 11
C_V = 2
min_pixels = 610000
if not os.path.exists(work_path):
    os.makedirs(work_path)
    os.makedirs(work_path + '_p/')
#img_path = 'F:\\project\\bottlecap\\20240217_outside\\red16500\\' #'F:\\project\\bottlecap\\Samples\\' + color_t + "\\"
#img_path = 'F:\\project\\bottlecap\\20240217_outside\\blue16500\\' #'F:\\project\\bottlecap\\Samples\\' + color_t + "\\"
#img_path = 'F:\\project\\bottlecap\\20240217_outside\\green12000\\' #'F:\\project\\bottlecap\\Samples\\' + color_t + "\\"
#img_path = 'F:\\project\\bottlecap\\20240217_outside\\white3900\\' #'F:\\project\\bottlecap\\Samples\\' + color_t + "\\"
# img_path = 'F:\\project\\bottlecap\\Samples\\blue\\' #'F:\\project\\bottlecap\\Samples\\' + color_t + "\\"
# img_path = 'F:\\project\\bottlecap\\SAMPLES OUTSIDE\\0326\\pink\\' #'F:\\project\\bottlecap\\Samples\\' + color_t + "\\"
img_path = 'F:\\project\\bottlecap\\test1\\in\white\\2024-07-01\\2\\'
# img_files = os.listdir(img_path)
img_files = [_ for _ in os.listdir(img_path) if (_.endswith('.jpg') or _.endswith('.png'))]
for img_file in img_files:
    prev_time = time.time()
    print(img_file)
    frame = cv.imread(img_path+img_file)
    frame = cv.addWeighted(frame, 1, frame, 0, 0)
    cv.imshow("source", frame)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    cv.imshow("HSV", hsv)

    if color_t == 'red':
        blur = 13
        thres = 23#24
        minSize = 50  * rate
        blockSize = 9
        C_V = 9
        hmin = 0          #0  # 156
        hmax = 15         #15   #20   #30   #181  # 180
        smin = 73          #140  #80   #20  # 43
        smax = 255         #255  #235  # 255
        vmin = 60          #90  #46  # 46
        vmax = 220         #255  #226  # 255
    elif color_t == 'blue':
        blur = 7
        thres = 12            #19  # 35  #23
        blockSize = 9
        C_V = 9
        minSize = 85 * rate
        # 藍
        hmin = 102         #100  # 109
        hmax = 180         #144  # 124
        smin = 100         #43
        smax = 255         #255
        vmin = 46          #46
        vmax = 255          #255
    elif color_t == 'gold':
        blur = 11
        thres = 40  # 23
        blockSize = 3
        C_V = 9
        minSize = 85 * rate
        hmin = 15  # 156
        hmax = 70  #190  # 180
        smin = 43   #15  # 43
        smax = 235 #207  #235  # 255
        vmin = 30  # 46
        vmax = 255  # 255
    elif color_t == 'green':
        blur = 15
        thres = 44
        blockSize = 11
        C_V = 7
        minSize = 100 * rate
        hmin = 3#30          #35  # 35
        hmax = 95#165         #108  # 77
        smin = 32#26          #43  # 43
        smax = 255#240         #220  # 255
        vmin = 37#61#46          #42  # 46
        vmax = 255#210         #225  # 255
    elif color_t == 'pink':
        blur = 7
        thres = 25  #64
        minSize = 50 * rate
        hmin = 68          #0  # 5
        hmax = 255         #212  # 180
        smin = 45         #5  # 43
        smax = 235         #245  # 215  #255
        vmin = 65         #38  # 46
        vmax = 255         #245  # 255
    elif color_t == 'lightblue':
        blur = 11       #15
        thres = 25      #40  #70  # 20
        minSize = 45 * rate
        hmin = 78          #78
        hmax = 120         #120  # 108
        smin = 71          #43
        smax = 248         #238  # 255
        vmin = 67          #67  # 46
        vmax = 255         #255  # 255
    elif color_t == 'white':
        blur = 7  # 5
        thres = 100#60  # 72 #112 #88
        minSize = 4 * rate
        max_Area = 9000
        blockSize = 9
        C_V = 3
        hmin = 0
        hmax = 221#180  # 140
        smin = 0
        smax = 31#34#53  # 35
        vmin = 54#22  # 200
        vmax = 220
    elif color_t == 'trans':
        blur = 7
        thres = 10       #23  #58  # 19 #66
        minSize = 40 * rate
        max_Area = 7000
        blockSize = 9
        C_V = 10
        hmin = 0           #0
        hmax = 221         #180  # 140
        smin = 0           #0
        smax = 100         #53
        vmin = 67          #22
        vmax = 255         #200
    else:  # black
        blur = 11
        thres = 128  # 240
        minSize = 85  * rate
        # color_t = color_t + '1'
        hmin = 0
        hmax = 36   #48  # 140
        smin = 0
        smax = 255
        vmin = 55   #0
        vmax = 98   #90  # 46
        min_pixels = 0

    hsv_low = np.array([hmin, smin, vmin])
    hsv_high = np.array([hmax, smax, vmax])
    mask = cv.inRange(hsv, hsv_low, hsv_high)
    frame_res = cv.bitwise_and(frame, frame, mask=mask)

    img_bin, img_o7, img_o7_inv = process_img(frame_res)

    cv.imwrite('temp_o/'+color_t+ '_p/'+ img_file + '_b.png', img_bin)
    cv.imwrite('temp_o/' + color_t + '_p/' + img_file + '_o7.png', img_o7)
    cv.imwrite('temp_o/'+color_t+ '_p/'+ img_file + '_o7_inv.png', img_o7_inv)
    img_res = process_blob(img_file, frame, img_bin, img_o7_inv)
    white_pixels = cv.countNonZero(img_bin)
    print('pixels=', white_pixels)

    if len(img_res) > 0:
        cv.imwrite('temp_o/'+color_t+'/result_' + img_file + '.png', img_res)
    else:
        white_pixels = cv.countNonZero(img_bin)
        print(white_pixels)
        if white_pixels < min_pixels:#600000: #800000:
            cv.imshow("all_zero", frame_res)
            cv.imwrite('temp_o/'+ color_t +'/result_' + img_file, frame_res)
    cv.waitKey(1)
    now_time = time.time()
    fps = (1 / (now_time - prev_time))
    print("FPS: {:.3f}".format(fps))
    # debug
    cv.waitKey(0)
    # debug


cv.waitKey(0)
cv.destroyAllWindows()