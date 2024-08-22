"""
BLOB特征分析(simpleblobdetector使用)
"""

import cv2 as cv
import numpy as np
import os
import time
import xml.etree.ElementTree as ET

def set_blob_param(category,para_name):
    param_file = ET.parse(para_name)
    root = param_file.getroot()
    id = 0
    param = []
    while root[0][id].tag != category:
        id += 1
    for i in range(0,15):
        param.append(int(root[0][id][i].attrib['value']))

    return param

def process_img(frame, mask, nW, nH):
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
        binary = cv.bitwise_and(binary, mask) #240820

        #lower = np.array([6, 6, 6])  # 轉換成 NumPy 陣列，範圍稍微變小 ( 55->30, 70->40, 252->200 )
        #upper = np.array([27, 27, 25])  # 轉換成 NumPy 陣列，範圍稍微加大 ( 70->90, 80->100, 252->255 )
        # lower = np.array([0, 0, 55])  # 轉換成 NumPy 陣列，範圍稍微變小 ( 55->30, 70->40, 252->200 )
        # upper = np.array([36, 255, 98])  # 轉換成 NumPy 陣列，範圍稍微加大 ( 70->90, 80->100, 252->255 )
        lower = np.array([hmin, smin, vmin])  # 轉換成 NumPy 陣列，範圍稍微變小 ( 55->30, 70->40, 252->200 )
        upper = np.array([hmax, smax, vmax])  # 轉換成 NumPy 陣列，範圍稍微加大 ( 70->90, 80->100, 252->255 )
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
        binary = cv.bitwise_and(binary, mask)  # 240820

    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    black = cv.imread('black1024.jpg')
    black = cv.resize(black, (nW, nH))
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
    r = param[13]
    cx = int(nW/2)  #512
    cy = int(nW/2)  #512
    cx1 = cx    #512
    cy1 = cy    #512

    for cnt in range(len(contours)):
        (x, y), radius = cv.minEnclosingCircle(contours[cnt])
        center = (int(x), int(y))
        radius = int(radius)
        if radius < (r-6):  #450:
            continue
        # debug
        print('radius=',radius)
        # debug
        if radius > (r+6):
            radius = param[13]
        if radius > r:
            r = radius
        # debug
        print('r=', r, 'r_m=', radius)
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
        y_min = nH  #1024
        x_min = nW  #1024
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
        # cv.circle(black, (int((cx1 + cx) / 2), cy), radius, (255, 255, 255), -1)

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

    # output7_inv = cv.bitwise_and(binary, output5) #(output5, output7)
    output7_inv = cv.bitwise_and(binary, gray2)  # (output5, output7)
    #output7_inv = cv.bitwise_or(binary, output5)
    # if color_t == 'trans':
    #    output7_inv = cv.bitwise_not(output7_inv)
    output7_inv = cv.bitwise_and(output7_inv, black_b)

    cv.imshow("O7_inv", output7_inv)

    ret, black_a = cv.threshold(binary, 1, 255, cv.THRESH_BINARY)
    black = cv.imread('black1024.jpg')
    black = cv.resize(black, (nW, nH))
    black_a = cv.cvtColor(black, cv.COLOR_BGR2GRAY)
    ret, black_a = cv.threshold(black_a, 1, 255, cv.THRESH_BINARY)
    diff = cv.absdiff(black_a, binary)
    ret, diff = cv.threshold(diff, 127, 255, cv.THRESH_BINARY_INV)
    cv.imshow("diff", diff)
    print('r=',r)
    #cv.circle(diff, (cx, cy), r-10, (0, 0, 0), 10)
    #cv.circle(diff, (int((cx1 + cx) / 2), cy), radius, (0, 0, 0), 1)
    # cv.circle(diff, (cx, cy), r, (0, 0, 0), 2)
    cv.circle(diff, (int((cx1 + cx) / 2), cy), r - 10, (0, 0, 0), param[14])
    # cv.circle(diff, (int((cx1 + cx) / 2), cy), r - 7, (0, 0, 0), 5)
    ret, diff = cv.threshold(diff, 127, 255, cv.THRESH_BINARY_INV)
    cv.imshow("diff1", diff)

    # debug
    # cv.waitKey(0)
    # debug
    #1 return binary, output7, output7_inv
    #240822 return diff, output7, binary
    return diff, output7, binary, (int((cx1 + cx) / 2), cy)

def process_blob(img,frame, binary, output7_inv):
    #edges_process(frame, output7_inv)

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
            cv.drawContours(frame, contours, cnt, (0, 0, 0), -1)
            print(cnt,':', area)
        else:
            cv.drawContours(frame, contours, cnt, (255, 0, 255), 1)
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
        result_i = cv.drawKeypoints(
            img, keypoints, blank, (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

#        for marker in keypoints:
#            cv.circle(frame, (int(marker.pt[0]), int(marker.pt[1])), 3, (255, 0, 255), -1, 8)
#            # cv.circle(frame, (int(marker.pt[0]), int(marker.pt[1])), int(marker.size/2), (0, 0, 255), 2, 1)
#            print('size:', marker.size)
#
#            result = cv.drawMarker(frame, tuple(int(i) for i in marker.pt), color=(0, 255, 0))
        cv.putText(result, str(len(keypoints)), (5, 20), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1)
        cv.putText(result_i, str(len(keypoints)), (5, 20), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1)
        cv.imshow("result", result_i)
        # cv.imwrite('temp_o/'+ color_t +' /result_' + file, result)
    return result

#240822 def edges_process(image, gray):
def edges_process(image, gray, center):
    max_distance = 40
    min_distance = param[13] - 25  #240822 3
    min_slope = 0.1

    edges = cv.Canny(gray, 50, 150)

    contours,_ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour_img = np.zeros_like(image)

    cv.drawContours(contour_img, contours, -1, (255, 255, 255), -1)
    cv.imshow("drawC0", contour_img)

    # gray = cv.cvtColor(contour_img, cv.COLOR_BGR2GRAY)
    cv.imshow("drawC1", gray)

    corners = cv.cornerHarris(gray, 2, 3, 0.04)
    print(corners.shape, corners.max())
    threshold = 0.5*corners.max()
    corners = cv.dilate(corners, None)

    #image[corners > threshold] = [0, 255, 0]
    contour_img[corners > threshold] = [0, 0, 255]
    cv.imshow("drawC2", contour_img)

    corner_points = np.argwhere(corners > threshold)
    #處理剔除相近的點
    for i in range(len(corner_points) - 1, 0, -1):
        #240822 if corner_points[i][0] == corner_points[i - 1][0]:
        if abs(corner_points[i][0] - corner_points[i - 1][0]) < 3 and abs(corner_points[i][1] - corner_points[i - 1][1]) < 3:
            corner_points = np.delete(corner_points, i, 0)
    #for i in range(len(corner_points) - 1, 0, -1):
    ##240822    if corner_points[i][1] == corner_points[i - 1][1]:
    #    if abs(corner_points[i][0] - corner_points[i - 1][0]) < 3 and abs(corner_points[i][1] - corner_points[i - 1][1]) < 3:
    #        corner_points = np.delete(corner_points, i, 0)

    #print(corner_points)

    slopes=[]
    slopes_point = []
    for i in range(len(corner_points)):
        (x2, y2) = center
        y1, x1 = corner_points[i]
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if distance >= min_distance:
            print(distance, end=' ')
            if x2 - x1 == 0:
                slope = float('inf')
            elif y2 - y1 == 0:
                slope = float('-inf')
            else:
                slope = (y2 - y1) / (x2 - x1)
                print(abs(slope), corner_points[i])
                #if abs(slope) < min_slope:
                #    break
            slopes.append(slope)
            slopes_point.append(corner_points[i])

#240822        #for j in range(i+1, len(corner_points)):
#240822        j = i+1
#240822        if j < len(corner_points):
#240822            y1, x1 = corner_points[i]
#240822            y2, x2 = corner_points[j]
#240822            distance = np.sqrt((x2-x1)**2+(y2-y1)**2)
#240822            # print(distance,end=', ')
#240822            #if distance < max_distance and distance >= min_distance:
#240822            if distance >= min_distance:
#240822                if x2 - x1 == 0:
#240822                    slope = float('inf')
#240822                elif y2 - y1 == 0:
#240822                    slope = float('-inf')
#240822                else:
#240822                    slope = (y2-y1)/(x2-x1)
#240822                    print(abs(slope), corner_points[i])
#240822                    if abs(slope) < min_slope:
#240822                        break
#240822                    # print(corner_points[i], corner_points[j])
#240822                    # cv.line(contour_img, (x1,y1), (x2,y2), (255, 0, 0), 1)
#240822                    # draw Cross
#240822                    #cv.line(contour_img, (x1-5, y1), (x1+5, y1), (255, 0, 0), 1)
#240822                    #cv.line(contour_img, (x1, y1-5), (x1, y1+5), (255, 0, 0), 1)
#240822                    #cv.imshow("drawC2", contour_img)
#240822
#240822                slopes.append(slope)
#240822                slopes_point.append(corner_points[i])
        #cv.circle(contour_img, (x1,y1), 5, [0, 255, 0], 1)
    print()
    #處理剔除相同的點
    i = 0
    for i in range(len(slopes_point) - 1, 0, -1):
        #240822 if slopes_point[i][0] == slopes_point[i - 1][0] and slopes_point[i][1] == slopes_point[i - 1][1]:
        if abs(slopes_point[i][0] - slopes_point[i - 1][0]) < 3 and abs(slopes_point[i][1] - slopes_point[i - 1][1]) < 3:
            slopes_point = np.delete(slopes_point, i, 0)
            slopes = np.delete(slopes, i, 0)
    # draw cross
    for i in range(len(slopes_point)):
        cv.line(contour_img, (slopes_point[i][1] - 5, slopes_point[i][0]), (slopes_point[i][1] + 5, slopes_point[i][0]), (255, 0, 0), 2)
        cv.line(contour_img, (slopes_point[i][1], slopes_point[i][0] - 5), (slopes_point[i][1], slopes_point[i][0] + 5), (255, 0, 0), 2)
        cv.imshow("drawC2", contour_img)

    slopes = [slope for slope in slopes if slope != float('inf') and slope != float('-inf')]
    slopes.sort()

    if len(slopes) >= 2:
        max_slope = max(abs(slopes[-2]), abs(slopes[1]))
        min_slope = min(abs(slopes[-2]), abs(slopes[1]))
    else:
        max_slope = float('-inf')
        min_slope = float('inf')
    print('Corners=', len(corner_points), 'Slope cnt=', len(slopes))
    print('max slope=', max_slope)
    print('min slope=', min_slope)
    #cv.putText(contour_img, str(len(corner_points))+' '+str(len(slopes)), (5, 20), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1)
    #cv.imshow('Contours_IMAGE', contour_img)

    return contour_img, corner_points, slopes_point


# img_path = 'F:\\project\\bottlecap\\SAMPLES OUTSIDE\\0326\\pink\\' #'F:\\project\\bottlecap\\Samples\\' + color_t + "\\"
img_path = 'D:\\project\\bottlecap\\test1\\in\\2024-07-22\\2\\resultNG\\' # 'D:\\project\\bottlecap\\test1\\out\\0717\\'
# img_files = os.listdir(img_path)

color_t = 'white'
work_path = 'temp_o/'+color_t
sens = 8

param = set_blob_param(color_t, 'Settings/oblob-param-newcam.xml') # add D:/project/bottlecap/code/
max_Area = param[3]
minSize = param[2] * (11 - sens) / 5
# rate = (11 - sens) / 5

blockSize = param[4]
C_V = param[5]
min_pixels = param[12]  #610000
blur = param[0]
thres = param[1]
C_V = param[5]
hmin = param[6]
hmax = param[7]
smin = param[8]
smax = param[9]
vmin = param[10]
vmax = param[11]

if not os.path.exists(work_path):
    os.makedirs(work_path)
    os.makedirs(work_path + '_p/')
img_files = [_ for _ in os.listdir(img_path) if (_.endswith('.jpg') or _.endswith('.png'))]
index = 0
while True:
# for img_file in img_files:
    prev_time = time.time()
    img_file = img_files[index]
    print(img_file)
    frame = cv.imread(img_path+img_file)
    frame = cv.addWeighted(frame, 1, frame, 0, 0)
    cv.imshow("source", frame)

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    cv.imshow("HSV", hsv)

    hsv_low = np.array([hmin, smin, vmin])
    hsv_high = np.array([hmax, smax, vmax])
    mask = cv.inRange(hsv, hsv_low, hsv_high)
    cv.imshow('mask', mask)
    frame_res = cv.bitwise_and(frame, frame, mask=mask)

    #240822 img_bin, img_o7, img_o7_inv = process_img(frame_res, mask, frame.shape[1], frame.shape[0])
    img_bin, img_o7, img_o7_inv, center = process_img(frame_res, mask, frame.shape[1], frame.shape[0])

    #極座標處理
    polar_img = cv.linearPolar(img_o7_inv, (img_o7.shape[1]//2, img_o7.shape[0]//2), img_o7.shape[1]//2+10, cv.WARP_FILL_OUTLIERS)
    edges =cv.Canny(polar_img, 50, 150)
    #cv.imwrite('temp_o/'+color_t+ 'polar'+ img_file + '_b.png', polar_img)
    #cv.imwrite('temp_o/'+color_t+ 'edges'+ img_file + '_b.png', edges)

    # edges_process(frame_res, img_o7_inv)

    cv.imwrite('temp_o/'+color_t+ '_p/'+ img_file + '_b.png', img_bin)
    cv.imwrite('temp_o/' + color_t + '_p/' + img_file + '_o7.png', img_o7)
    cv.imwrite('temp_o/'+color_t+ '_p/'+ img_file + '_o7_inv.png', img_o7_inv)
    img_res = process_blob(frame, frame_res, img_bin, img_o7_inv)
#-----------------------------------------------------------------------
    #240822 contour_img, corner_points, slopes_pt = edges_process(frame, img_o7_inv)
    contour_img, corner_points, slopes_pt = edges_process(frame, img_o7_inv, center)
#-----------------------------------------------------------------------
    white_pixels = cv.countNonZero(img_bin)
    print('pixels=', white_pixels)

    if len(img_res) > 0:
        cv.putText(img_res, str(len(corner_points)) + ' ' + str(len(slopes_pt)), (5, 40), cv.FONT_HERSHEY_PLAIN, 1.2,
                   (0, 255, 0), 1)
        for i in range(len(slopes_pt)):
            cv.circle(img_res, (slopes_pt[i][1], slopes_pt[i][0]), 10, [0, 0, 255], 1)
        cv.imshow('result_all', img_res)
        cv.imwrite('temp_o/'+color_t+'/result_' + img_file + '.png', img_res)
    else:
        white_pixels = cv.countNonZero(img_bin)
        print(white_pixels)
        if white_pixels < min_pixels:#600000: #800000:
            cv.putText(frame_res, str(len(corner_points)) + ' ' + str(len(slopes_pt)), (5, 40), cv.FONT_HERSHEY_PLAIN,
                       1.2,
                       (0, 255, 0), 1)
            for i in range(len(slopes_pt)):
                cv.circle(frame_res, (slopes_pt[i][1], slopes_pt[i][0]), 10, [0, 0, 255], 1)
            cv.imshow('result_all', frame_res)
            cv.imshow("all_zero", frame_res)
            cv.imwrite('temp_o/'+ color_t +'/result_' + img_file, frame_res)
    now_time = time.time()
    fps = (1 / (now_time - prev_time))
    key = cv.waitKeyEx(0)
    if key & 0xff == ord('q'):
        break
    elif key == 2490368 or key == 2424832:
        index -= 1;
        if index < 0:
            index = len(img_files)-1
    elif key == 2621440 or key == 2555904:
        index += 1;
        if index >= len(img_files):
            index = 0
    print("FPS: {:.3f}".format(fps))
    # debug
    # cv.waitKey(0)
    # debug


# cv.waitKey(0)
cv.destroyAllWindows()