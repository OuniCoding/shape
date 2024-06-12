"""
BLOB特征分析(simpleblobdetector使用)
"""

import cv2 as cv
import numpy as np
import os
import time

#img_path = 'F:\\project\\bottlecap\\20240217_outside\\red16500\\' #'F:\\project\\bottlecap\\Samples\\' + color_t + "\\"
#img_path = 'F:\\project\\bottlecap\\20240217_outside\\blue16500\\' #'F:\\project\\bottlecap\\Samples\\' + color_t + "\\"
#img_path = 'F:\\project\\bottlecap\\20240217_outside\\green12000\\' #'F:\\project\\bottlecap\\Samples\\' + color_t + "\\"
#img_path = 'F:\\project\\bottlecap\\20240217_outside\\white3900\\' #'F:\\project\\bottlecap\\Samples\\' + color_t + "\\"
img_path = 'F:\\project\\bottlecap\\Samples\\black\\' #'F:\\project\\bottlecap\\Samples\\' + color_t + "\\"
#img_path = 'F:\\project\\bottlecap\\SAMPLES OUTSIDE\\0326\\tr\\' #'F:\\project\\bottlecap\\Samples\\' + color_t + "\\"
# img_files = os.listdir(img_path)
#img_path = 'F:\\project\\bottlecap\\test1\\0529\\red\\'
img_path = 'F:\\project\\bottlecap\\test1\\in\\black\\2024-06-05\\1\\resultG\\'
#img_path = 'F:\\project\\bottlecap\\test1\\0530\\Logo\\red\\2024-05-30\\1\\resultNG\\'


max_Area = 5000
color_t = 'black'
work_path = 'temp/'+color_t
minSize = 85
blockSize = 13
C_V = 2
min_pixels = 800000

if not os.path.exists(work_path):
    os.makedirs(work_path)
    os.makedirs(work_path + '_p/')

def process_img(frame):
    cv.imshow("input", frame)
    #gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
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
    if color_t == 'black':
        ret, binary = cv.threshold(gray2, thres, 255, cv.THRESH_BINARY)# + cv.THRESH_OTSU)
        lower = np.array([0, 0, 55])  # 轉換成 NumPy 陣列，範圍稍微變小 ( 55->30, 70->40, 252->200 )
        upper = np.array([36, 255, 98])  # 轉換成 NumPy 陣列，範圍稍微加大 ( 70->90, 80->100, 252->255 )
        output = cv.inRange(frame, lower, upper)   # 取得顏色範圍的顏色
        cv.imshow("O", output)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (blur, blur))  # 設定膨脹與侵蝕的參數
        output = cv.dilate(output, kernel)       # 膨脹影像，消除雜訊
        output = cv.erode(output, kernel)        # 縮小影像，還原大小
        output9 = cv.bitwise_and(gray2, gray, mask=output)
        #cv.imshow("O9", output9)
        ret, output9 = cv.threshold(output9, 7, 255, cv.THRESH_BINARY_INV)
        cv.imshow("O9", output9)

        cv.imshow("BIN", binary)
        output5 = cv.adaptiveThreshold(binary, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize, C_V)
        cv.imshow("O5", output5)
        output7 = cv.subtract(output5, gray2)
        # ret, output7 = cv.threshold(output7, 250, 255, cv.THRESH_BINARY_INV)
        cv.imshow("O7", output7)
        output8 = cv.subtract(output5, output7)
        cv.imshow("O8", output8)

        output7_inv = cv.bitwise_or(binary, output5)   #(binary, gray2) #
        #output7_inv = cv.subtract(binary, gray2) #(output5, output7)
        # if color_t == 'trans':
        #    output7_inv = cv.bitwise_not(output7_inv)

        cv.imshow("O7_inv", output7_inv)


        return binary, output7, output9 #output7_inv # output #,

    else:
        ret, binary = cv.threshold(gray2, thres, 255, cv.THRESH_BINARY)  # + cv.THRESH_OTSU)


    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    black = cv.imread('black1120.jpg')
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
        if radius < 500:
            continue
        if radius > 535:
            radius = 521
        #cv.circle(black, center, radius,(255, 255, 255), -1)
        print(radius)
        # 提取與繪制輪廓
        #cv.drawContours(frame, contours, cnt, (0, 255, 0), -1)
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
        y_min = 1120
        x_min = 1120
        y = 0
        x = 0
        dis_max = 0
        for c in range(corners):
            [xy] = approx[c]
            y = xy[1:2]
            x = xy[:1]
            # cv.circle(black, (int(x), int(y)), 5, (0, 255, 0), -1)
            # cv.line(black, (int(x), int(y)), (cx, cy), (0, 0, 255), 1)
            if y > y_max:       # x > x_max :  #
                y_max = int(y)
                x_max = int(x)
                [xy_previous] = approx[c-1]
            if y < y_min:
                y_min = int(y)
                x_min = int(x)

        dis = int((((x_max - x_min) ** 2 + (y_max - y_min) ** 2) ** 0.5) / 2 * 1)
        print(x_max, y_max, x_min, y_min, dis)
        d_x = abs((x_max - x_min) / 2)
        d_y = abs((y_max - y_min) / 2)
        if x_max > x_min:
            cx1 = int(x_min + d_x)
        else:
            cx1 = int(x_max + d_x)
        cy1 = int(y_min + d_y)
        print('cx1=', cx1, ' cy1=', cy1)
        print('cx=', cx, 'cy=', cy)
        if (x_max > cx and x_min > cx) or (x_max < cx and x_min < cx):
            cx1 = cx
        #if radius > dis: radius = dis

        cv.circle(black, (cx, cy), radius, (255, 255, 255), -1)
        #cv.circle(black, (cx1, cy1), radius, (255, 255, 255), -1)
        cv.circle(black, (cx, cy), 5, (0, 255, 0), -1)
        cv.circle(black, (cx1, cy1), 2, (0, 0, 255), -1)
        # cv.circle(black, (cx1, cy1), dis, (0, 0, 255), 1)
        # cv.imshow('line', black)
        # cv.waitKey(0)

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
    #binary = cv.bitwise_and(binary, black_b)

    cv.imshow("BIN", binary)
    output5 = cv.adaptiveThreshold(gray2, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize, C_V)
    cv.imshow("O5", output5)
    output7 = cv.subtract(output5, gray2)
    cv.imshow("O7", output7)
    output8 = cv.subtract(output5, output7)
    cv.imshow("O8", output8)

    output7_inv = cv.bitwise_or(binary, gray2)
    #output7_inv = cv.bitwise_and(binary, output5)
    # if color_t == 'trans':
    #    output7_inv = cv.bitwise_not(output7_inv)
    output7_inv = cv.bitwise_and(output7_inv, black_b)

    cv.imshow("O7_inv", output7_inv)

    return binary, output7, output7_inv

def process_blob(file, frame, binary, output7_inv):
    params = cv.SimpleBlobDetector_Params()

    # change thresholds
    params.minThreshold = 0
    params.maxThreshold = 200   #120

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
    params.filterByConvexity = True
    params.minConvexity = 0.01

    # Filter by Inertia
    params.filterByInertia = False  #True
    params.minInertiaRatio = 0.01    #0.5

    # contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # contours, hierarchy = cv.findContours(output9, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv.findContours(output7_inv, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in range(len(contours)):
        area = cv.contourArea(contours[cnt])
        # debug
        # print(cnt, ':', area)
        # debug
        if area >= minSize and area <= max_Area:     # trans:7000
            # 提取與繪制輪廓
            #cv.drawContours(frame, contours, cnt, (0, 255, 0), 1)
            #cv.imshow("contours", frame)
            print(cnt,':', area)
            #debug cv.waitKey(1)
        #else:
        #    cv.drawContours(frame, contours, cnt, (0, 25, 255), 1)
        #   cv.imshow("contours", frame)
    # 提取关键点
    keypoints = []
    detector = cv.SimpleBlobDetector_create(params)
    keypoints = detector.detect(output7_inv)    #(binary) #  gray
    #keypoints = detector.detect(binary) #  gray
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
        # cv.imwrite('temp/'+ color_t +'/result_' + file, result)

    return result

img_files = [_ for _ in os.listdir(img_path) if (_.endswith('.jpg') or _.endswith('.png'))]
for img_file in img_files:
    prev_time = time.time()
    print(img_file)
    frame = cv.imread(img_path+img_file)
    frame = cv.addWeighted(frame, 1, frame, 0, 0)
    cv.imshow("source", frame)
    #frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    cv.imshow("HSV", hsv)

    if color_t == 'red':
        blur = 13    #13
        thres = 23  #60  # 23
        minSize = 50
        blockSize = 13
        C_V = 2
        hmin = 0    #0  # 156
        hmax = 255   #11  # 180
        smin = 75   #0  # 43
        smax = 255  #255  # 255
        vmin = 60   #46  # 46
        vmax = 220     #255  # 255
    elif color_t == 'blue':
        blur = 11#15
        thres = 10#23
        blockSize = 9   #11
        C_V = 9 #5
        # 藍
        hmin = 105
        hmax = 200
        smin = 62#72
        smax = 255
        vmin = 35#26
        vmax = 205
    elif color_t == 'gold':
        blur = 11
        thres = 20    #65  #40  # 23
        blockSize = 3   #13
        C_V = 9#2
        minSize = 45
        hmin = 5      #5    # 156
        hmax = 37     #178  #190  # 180
        smin = 50     #0    # 43
        smax = 228    #200#207#247  #235  # 255
        vmin = 60#40     #0    # 46
        vmax = 235 #255  # 255
    elif color_t == 'green':
        blur = 11#13#11     #15
        thres = 12#11        #44
        blockSize = 13
        C_V = 2
        minSize = 45
        hmin = 40#30
        hmax = 128#99
        smin = 26
        smax = 240
        vmin = 44#46
        vmax = 210
    elif color_t == 'pink':
        blur = 5   #7
        thres = 25    #81  #26  #64
        minSize = 50
        blockSize = 13
        C_V = 2
        hmin = 68     #0  # 5
        hmax = 255     #178  #212  # 180
        smin = 45      #0    #5  # 43
        smax = 255     #255  #245  # 215  #255
        vmin = 65      #0    #38  # 46
        vmax = 255     #255  #245  # 255
    elif color_t == 'lightblue':
        blur = 11
        thres = 25#11
        minSize = 50
        hmin = 90#70
        hmax = 150#190
        smin = 45#71
        smax = 248
        vmin = 65#20
        vmax = 255
    elif color_t == 'white':
        blur = 3#7
        thres = 100#69
        minSize = 45
        max_Area = 9000
        blockSize = 9
        C_V = 3
        hmin = 0
        hmax = 221
        smin = 0
        smax = 45#100
        vmin = 67
        vmax = 255
    elif color_t == 'trans':
        blur = 5#7
        thres = 10#5
        minSize = 50
        max_Area = 7000
        blockSize = 9
        C_V = 13
        hmin = 0
        hmax = 221
        smin = 0
        smax = 50#100
        vmin = 67
        vmax = 255
    else:  # black
        blur = 11
        thres = 128 # 240
        #color_t = color_t + '1'
        hmin = 0
        hmax = 36   #48  # 140
        smin = 0
        smax = 255
        vmin = 55   #0
        vmax = 98   #90  # 46
        min_pixels = 0

    if color_t == 'black':
        frame_res = frame
    else:
        hsv_low = np.array([hmin, smin, vmin])
        hsv_high = np.array([hmax, smax, vmax])
        mask = cv.inRange(hsv, hsv_low, hsv_high)
        frame_res = cv.bitwise_and(frame, frame, mask=mask)
    #if color_t == 'red':
    #    hmin = 100  # 156
    #    hmax = 255  # 180
    #    smin = 0  # 43
    #    smax = 255  # 255
    #    vmin = 46  # 46
    #    vmax = 255  # 255
    #    hsv_low = np.array([hmin, smin, vmin])
    #    hsv_high = np.array([hmax, smax, vmax])
    #    mask = cv.inRange(hsv, hsv_low, hsv_high)
    #    res2 = cv.bitwise_and(frame, frame, mask=mask)
    #    frame_res = cv.bitwise_or(frame_res, res2)

    img_bin, img_o7, img_o7_inv = process_img(frame_res)

    cv.imwrite('temp/'+color_t+ '_p/'+ img_file + '_b.png', img_bin)
    cv.imwrite('temp/' + color_t + '_p/' + img_file + '_o7.png', img_o7)
    cv.imwrite('temp/'+color_t+ '_p/'+ img_file + '_o7_inv.png', img_o7_inv)
    img_res = process_blob(img_file, frame_res, img_bin, img_o7_inv)
    white_pixels = cv.countNonZero(img_bin)
    print('pixels=', white_pixels)
    if len(img_res) > 0:
        cv.imwrite('temp/'+color_t+'/result_' + img_file + '.png', img_res)
        cv.waitKey(0)
    else:
        if white_pixels < min_pixels:
            cv.imshow("all_zero", frame_res)
            cv.imwrite('temp/'+ color_t +'/result_' + img_file, frame_res)
    cv.waitKey(1)
    now_time = time.time()
    fps = (1 / (now_time - prev_time))
    print("FPS: {:.3f}".format(fps))
    # debug
    cv.waitKey(0)
    # debug

cv.waitKey(1)
cv.destroyAllWindows()