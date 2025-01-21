"""
BLOB特征分析(simpleblobdetector使用)
"""

import cv2 as cv
import numpy as np
import os
import time
import xml.etree.ElementTree as ET

diameter_v = 29
def set_blob_param(category,para_name):
    param_file = ET.parse(para_name)
    root = param_file.getroot()
    id = 0
    param = []
    while root[0][id].tag != category:
        id += 1

    for i in range(0, 20):
        if i < 17:
            param.append(int(root[0][id][i].attrib['value']))
        else:
            param.append(float(root[0][id][i].attrib['value']))
    param.append(int(root[0][id][20].attrib['value']))

    unit_p = diameter_v / 2 / param[13]
    if category != 'trans':
        param[2] = int(0.3 * 0.3 / (unit_p) ** 2) # min_Area

    return param

def read_path_color():
    param_file = 'param_out.ini'

    ini_file = open(param_file, 'r')
    color = ini_file.readline()
    color = color.replace('\n','')
    path = ini_file.readline()
    path = path.replace('\n','')

    ini_file.close()

    return color, path

def process_img(frame, mask, nW, nH):
    diameter_v = 29
    if param[16] == 1:
        err_max = param[17]
        err_min = param[18]
        er_pixel = int((err_max+err_min)/ 2 / 2 / (diameter_v / 2 / param[13]))
    else:
        err_max = 1.5
        err_min = 1.5
        er_pixel = int(1.5/2/(diameter_v/2/param[13]))
    err_rad = int(param[19])

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
    if color_t == 'gold':# or color_t == 'trans':
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
        cv.imshow("BI", binary)
        binary = cv.bitwise_and(binary, mask)  # 240820

    #contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
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
    if param[16] == 1:
        r_min = r - int(err_min / 2 / (diameter_v / 2 / param[13])) * (11 - defor) / 4
        r_max = r + int(err_max / 2 / (diameter_v / 2 / param[13])) * (11 - defor) / 4
    else:
        r_min = r - er_pixel * (11 - defor) / 4
        r_max = r + er_pixel * (11 - defor) / 4
    print(r_min, r_max)

    # 擬合圓形
    if len(contours):
        contour = max(contours, key=cv.contourArea)
        (x, y), radius = cv.minEnclosingCircle(contour)
    else:
        radius = param[13]
        x = int(nW / 2)  # 512
        y = int(nW / 2)  # 512

    center = (int(x), int(y))
    radius = int(radius)

    if radius > r:
        r = radius
    cv.circle(black, center, radius, (255, 255, 255), -1)

    ''' 241210 ------ off ------
    for cnt in range(len(contours)):
        (x, y), radius = cv.minEnclosingCircle(contours[cnt])
        center = (int(x), int(y))
        radius = int(radius)
        if radius < r_min:  #(r-er_pixel * (10-defor)/4):  #450:
            continue
        # debug
        print('radius=', radius)
        # debug
        if radius > r_max:  #(r+er_pixel* (10-defor)/4):   #60
            continue    # radius = param[13]
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
    ---- off -----'''
        # cv.circle(black, (int((cx1 + cx) / 2), cy), radius, (255, 255, 255), -1)
        #中心標註
        #cv.circle(black, (cx, cy), 5, (0, 255, 0), -1)
        #cv.circle(black, (cx1, cy1), 2, (0, 0, 255), -1)
        # 中心標註
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
    if color_t != 'black':
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
    #dd=diff.copy()
    #cv.circle(dd, (int((cx1 + cx) / 2), cy), r, (0, 0, 0), 1)
    cv.imshow("diff", diff)
    #cv.imshow("dd", dd)
    print('r=',r)
    #cv.circle(diff, (cx, cy), r-10, (0, 0, 0), 10)
    #cv.circle(diff, (int((cx1 + cx) / 2), cy), radius, (0, 0, 0), 1)
    # cv.circle(diff, (cx, cy), r, (0, 0, 0), 2)

    '''off 變形檢測
    if color_t != 'black':
        cv.circle(diff, (int((cx1 + cx) / 2), cy), r - er_pixel, (0, 220, 0), int(er_pixel * (11-defor) / 5)) #param[14]
    #cv.circle(diff, (int((cx1 + cx) / 2), cy), r, (0, 0, 0), int(er_pixel * (11-defor) / 5)) #param[14]
    '''
    print('er_pixel=', er_pixel)
    # cv.circle(diff, (int((cx1 + cx) / 2), cy), r - 7, (0, 0, 0), int(param[14] * (11-defor) / 5)) #param[14]
    # cv.circle(diff, (int((cx1 + cx) / 2), cy), r - 7, (0, 0, 0), 5)
    #diff = cv.bitwise_or(diff, dd)
    ret, diff = cv.threshold(diff, 127, 255, cv.THRESH_BINARY_INV)
    cv.imshow("diff0", diff)
    diff = cv.bitwise_and(diff, binary)  # 241202
    '''off 變形檢測
    if color_t != 'black':
        cv.circle(diff, (int((cx1 + cx) / 2), cy), (r - err_rad), (255, 255, 255), (11-defor))
    '''
    cv.imshow("diff1", diff)

    # debug
    # cv.waitKey(0)
    # debug
    #1 return binary, output7, output7_inv
    #240822 return diff, output7, binary
    # 241210 add "contours, center, radius"
    return diff, output7, binary, (int((cx1 + cx) / 2), cy), contours, center, radius

def process_blob(img,frame, binary, output7_inv, contours, center, radius):
    #edges_process(frame, output7_inv)
    err_max = param[17]
    err_min = param[18]
    er_pixel = int((err_max + err_min) / 2 / 2 / (diameter_v / 2 / param[13]))

    unit_p = 0.3 / (diameter_v / 2 / param[13])
    minSize_dia = minSize ** 0.5
    maxArea_dia = max_Area ** 0.5
#'''-----
    kp_findContour = []
    if color_t != 'black':
        # 確保只有一個主要輪廓，找到最大的輪廓
        if len(contours) == 0:
            result = []
            return result
# -----------------邊緣瑕疵檢測: 不規則圓弧與圓弧的缺口檢測
        ## 邊緣檢測
        #edges = cv.Canny(binary, 128, 255, apertureSize=5, L2gradient=True) #  apertureSize=3, L2gradient=False
        #
        ## 找到輪廓
        #contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        ## 找到最大的輪廓（假設為圓弧）
        #contour = max(contours, key=cv.contourArea)
        ##不規則圓弧缺口
        ## 使用 cv2.fitEllipse 擬合橢圓
        #if len(contour) >= 2:  # fitEllipse 要求輪廓點數 >= 5
        #    ellipse = cv.fitEllipse(contour)
        #    centerM, axes, angle = ellipse[0], ellipse[1], ellipse[2]
        #    result = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
        #    cv.ellipse(result, ellipse, (0, 255, 0), 2)  # 畫出擬合的橢圓
        #
        #    # 檢測缺口：計算每個點到橢圓的距離
        #    defects = []
        #    for point in contour:
        #        px, py = point[0]
        #        rx = (px - centerM[0]) * np.cos(np.radians(angle)) + (py - centerM[1]) * np.sin(np.radians(angle))
        #        ry = (py - centerM[1]) * np.cos(np.radians(angle)) - (px - centerM[0]) * np.sin(np.radians(angle))
        #        distance = (rx / (axes[0] / 2)) ** 2 + (ry / (axes[1] / 2)) ** 2
        #        if distance > 1.1 or distance < 0.9:  # 超出橢圓範圍
        #            defects.append((px, py))
        #
        #    # 標記缺口
        #    for defect in defects:
        #        cv.circle(result, defect, 5, (0, 0, 255), -1)  # 紅色標記缺口
        #
        #    # 顯示結果
        #    cv.imshow("Defects", result)
        #
        #
        ##圓弧缺口
        ## 擬合圓
        #(xM, yM), radiusM = cv.minEnclosingCircle(contour)
        #centerM = (int(xM), int(yM))
        #radiusM = int(radiusM)
        ## 初始化缺口點數組
        #defects = []
        ## 計算每個點到擬合圓的距離
        #for point in contour:
        #    px, py = point[0]
        #    distance = np.sqrt((px - xM) ** 2 + (py - yM) ** 2)
        #
        #    # 如果距離小於一定值，判定為缺口
        #    if abs(distance - radiusM) > 7:  # 10 為閾值，可根據需求調整
        #        defects.append((px, py))
        #
        ## 繪製結果
        #result = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
        #cv.circle(result, centerM, radiusM, (0, 255, 0), 2)  # 繪製擬合圓
        #
        ## 標記缺口
        #for defect in defects:
        #    cv.circle(result, defect, 5, (0, 0, 255), -1)  # 紅色標記缺口
        #cv.imshow("Defects1", result)


    #edges = cv.Canny(binary, 128, 255, apertureSize=5, L2gradient=True)  # apertureSize=3, L2gradient=False

        # 找到輪廓
        #contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # 找到最大的輪廓（假設為圓弧）
        contour = max(contours, key=cv.contourArea)
        # 不規則圓弧缺口
        # 使用 cv2.fitEllipse 擬合橢圓
        if len(contour) >= 3:  # fitEllipse 要求輪廓點數 >= 5
            ellipse = cv.fitEllipse(contour)
            centerM, axes, angle = ellipse[0], ellipse[1], ellipse[2]
            cv.ellipse(binary, ellipse, (255, 255, 255), param[20])  # 畫出擬合的橢圓
            cv.imshow("fitEllipse", binary)

        x, y = center
        print((x, y), radius)
#
#        result = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
#        cv.circle(result, center, radius, (0, 255, 0), 1)
#        cv.imshow("R", result)
#
#        # 繪製擬合圓形
#        cv.circle(binary, center, radius, (255, 255, 255), 1)
#        cv.imshow("Binary", binary)
#
#        # 檢測是否有缺陷（偏離圓形的部分）
#        for point in contour:
#            dist = cv.pointPolygonTest(contour, (x, y), True)
#            if abs(dist) >= (radius - er_pixel):  # 偏差超過5個像素，標記為缺陷
#                # cv.circle(binary, tuple(point[0]), int(abs(dist-radius)), (255, 255, 255), -1)
#                cv.circle(binary, tuple(point[0]), 1, (255, 255, 255), -1)
#                cv.circle(frame, tuple(point[0]), 1, (0, 0, 255), -1)
#        cv.imshow("Binary1", binary)
#
#        area = cv.contourArea(contour)
#        perimeter = cv.arcLength(contour, True)
#
#        # 圓形度公式：4 * π * (面積) / (周長)^2
#        circularity = 4 * np.pi * (area / (perimeter ** 2))
#        # 判斷是否接近圓形
#        if 0.5 < circularity < 1.3:  # 可調整範圍
#            cv.drawContours(binary, contour, -1, (255, 255, 255), 1)
#        cv.circle(binary, center, radius - 3, (255, 255, 255), int(param[19]+1))
#        cv.imshow("Binary2", binary)
#
#        # ''' off find contour --------
        contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # output = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)

        all_point = []
        all_area = 0
        keypoints = []

        result = []
        cx_p = 0
        cy_p = 0
        for cnt in range(len(contours)):
            # 求解面積
            area = cv.contourArea(contours[cnt])
            perimeter = cv.arcLength(contours[cnt], True)
            # print(cnt, ':', area)
            # if area >= 40 and area <= max_Area:  # trans:7000
            if area >= minSize and area <= max_Area:  # trans:7000
                # 求解中心位置
                mm = cv.moments(contours[cnt])
                cx = int(mm['m10'] / mm['m00'])
                cy = int(mm['m01'] / mm['m00'])
                if abs(cx - cx_p) < 5 and abs(cy - cy_p) < 5:
                    continue

                length = np.sqrt((cx - x) ** 2 + (cy - y) ** 2)
                print(f'{cnt}, area={area}, Length={length}, Perimeter = {perimeter}, ', end='')
                print(f", (座標): ({cx}, {cy})")
                if length <= radius:
                    keypoints.append(contours[cnt])
                else:
                    cv.drawContours(frame, contours, cnt, (0, 255, 0), 1)

                cx_p = cx
                cy_p = cy
            elif area < minSize and area > 10:
                # 求解中心位置
                mm = cv.moments(contours[cnt])
                cx = int(mm['m10'] / mm['m00'])
                cy = int(mm['m01'] / mm['m00'])

                print(f'Small {cnt}, area={area},(座標): ({cx}, {cy}) ')

                length = np.sqrt((cx - x) ** 2 + (cy - y) ** 2)
                if length <= radius:
                    if abs(cx - x) <= 20 and abs(cy - y) <= 20:
                        keypoints.append(contours[cnt])
                    all_point.append((cx, cy))
                    all_area = area + all_area
                cv.drawContours(frame, contours, cnt, (0, 255, 128), 1)
                # cv.drawContours(binary, contours, cnt, (255, 255, 255), -1)
            else:
                cv.drawContours(frame, contours, cnt, (255, 0, 0), 1)
                # cv.drawContours(binary, contours, cnt, (255, 255, 255), -1)

        for cnt in range(len(keypoints)):
            cv.drawContours(frame, keypoints, cnt, (0, 0, 255), 1)

        cv.imshow("contours", frame)

        kp_findContour = keypoints
        # binary = cv.cvtColor(output, cv.COLOR_BGR2GRAY)
        # cv.imshow("BinOut", binary)
        print(f'\nKeyPoints={len(keypoints)}, all small area={all_area}, all small points={len(all_point)}')
        #result = cv.putText(frame, str(len(keypoints)), (100, 20), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255), 1)
        result = cv.putText(frame, f'{len(keypoints)}  {len(all_point)} X small area={all_area}', (5, 20), cv.FONT_HERSHEY_PLAIN, 1.2,
                            (0, 255, 255), 1)
        cv.imshow("blob_result", result)
        # ------- off find contour '''
        return result
##------
    params = cv.SimpleBlobDetector_Params()

    # change thresholds
    params.minThreshold = 0     #0
    params.maxThreshold = 200   #120

    # 重复的最小次数，只有属于灰度图像斑点的那些二值图像斑点数量大于该值时，该灰度图像斑点才被认为是特征点
    params.minRepeatability = 2
    # 最小的斑点距离，不同二值图像的斑点间距离小于该值时，被认为是同一个位置的斑点，否则是不同位置上的斑点
    params.minDistBetweenBlobs = 10

    # filter by color
    params.filterByColor = False    #True     #
    params.blobColor = 0

    # filter by area
    params.filterByArea = True
    params.minArea = 10  # minSize    #50 #85
    params.maxArea = max_Area   #2000   trans:7000

    # filter by circularity 圓形度公式：4 * π * (面積) / (周長)^2
    params.filterByCircularity = False  #True
    params.minCircularity = 0.01
    params.maxCircularity = 150

    # Filter by Convexity
    params.filterByConvexity = True   #False  #True
    params.minConvexity = 0.1   #0.87
    params.maxConvexity = 10.0

    # Filter by Inertia
    params.filterByInertia = True   #False  #True
    params.minInertiaRatio = 0.01    #0.5
    params.maxInertiaRatio = 10.0

    # contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # contours, hierarchy = cv.findContours(output9, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #''' off find contour --------
    ''' 241210 off findContours, receive "contours, center, radius" from process_img
    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # 確保只有一個主要輪廓，找到最大的輪廓
    if len(contours) == 0:
        result = []
        return result
    contour = max(contours, key=cv.contourArea)

    # 擬合圓形
    (x, y), radius = cv.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)
    off findContours, receive "contours, center, radius" from process_img ----- '''
    #'''-------- off Blob Detector ----------
    # 提取关键点
    keypoints = []
    detector = cv.SimpleBlobDetector_create(params)
    if color_t != 'black':
        keypoints = detector.detect(binary) #(output7_inv)    #  gray
    #black else:
    #black     keypoints = detector.detect(binary) #(output7_inv)  # (binary) #  gray
    print('keypoints=', len(keypoints))
    result = []
    blank = np.zeros((1, 1))
    #if len(keypoints) > 100:
    #    kp = keypoints[0:100]
    #else:
    #    kp = keypoints
    kp = []
    small_points = 0
    all_point = []
    all_area = 0
    if len(keypoints) > 0:
        # result = frame.copy()
        result = cv.drawKeypoints(frame, keypoints, blank, (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #result = cv.drawKeypoints(frame, kp, blank, (0, 255, 255), cv.DRAW_MATCHES_FLAGS_DEFAULT)
        #result = cv.drawKeypoints(frame, kp, blank, (0, 255, 255), cv.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
        #result = cv.drawKeypoints(frame, kp, blank, (0, 255, 255), cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        result_i = cv.drawKeypoints(img, keypoints, blank, (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # if len(keypoints) > 0:
    #     result = cv.drawKeypoints(
    #         frame, keypoints, blank, (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    #     )
    #     result_i = cv.drawKeypoints(
    #         img, keypoints, blank, (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    #     )

        for marker in keypoints:   # keypoints:
            px = int(marker.pt[0])
            py = int(marker.pt[1])
            psize = marker.size
            if psize >= minSize_dia:
                if psize <= maxArea_dia:
                    # same_flag = False
                    # for i in range(len(kp_findContour)):
                    #     ix, iy = kp_findContour[i]
                    #     if px == ix and py == iy:

                    #        kp.append(marker)
                    #        # result = cv.circle(frame, (px, py), int(psize/2), (0, 0, 255), 1)
                    #        cv.putText(result, str('%.2f'%psize), (px + 10, py), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1)
                    #        print(f'Diameter:{marker.size}, (座標): ({px}, {py})')
                    #        same_flag = True
                    #        break
                    kp.append(marker)
                    cv.putText(result, str('%.2f' % psize), (px + 10, py), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1)
                    print(f'Diameter:{marker.size}, (座標): ({px}, {py})')
                    # if not same_flag:
                    #     cv.putText(result, str('%.2f' % psize), (px + 10, py), cv.FONT_HERSHEY_PLAIN, 1, (0, 100, 255),
                    #                1)
                    #     print(f'xxxDiameter:{marker.size}, (座標): ({px}, {py})')

                else:
                    cv.putText(result, str('%.2f' % psize), (px + 10, py), cv.FONT_HERSHEY_PLAIN, 1, (100, 255, 120), 1)
                    print(f'<= Diameter:{marker.size}, (座標): ({px}, {py})')
            else:   # < minSize
            #elif psize < unit_p:
                small_points += 1
                all_point.append((px, py))
                all_area = np.pi * (psize / 2) ** 2 + all_area
            # cv.circle(frame, (int(marker.pt[0]), int(marker.pt[1])), 3, (255, 0, 255), -1, 8)
#           # cv.circle(frame, (int(marker.pt[0]), int(marker.pt[1])), int(marker.size/2), (255, 0, 255), 1, 1)
                cv.putText(result, str('%.2f' % psize), (px, py), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                # print('**size:', marker.size)

        print('All keypoints =', len(kp) ,', all small points=', small_points)
        result = cv.drawKeypoints(result, kp, None, (0, 255, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
#            result = cv.drawMarker(frame, tuple(int(i) for i in marker.pt), color=(0, 255, 0))
        cv.putText(result, str(len(kp)), (5, 20), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1)
        cv.putText(result_i, str(len(keypoints)), (5, 20), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1)
        cv.imshow("blob_result", result)
        # cv.imwrite('temp_o/'+ color_t +' /result_' + file, result)
    #-------- off Blob Detector ----------'''
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
# img_path = 'f:\\project\\bottlecap\\test1\\out\\green\\2024-09-19\\2\\resultNG\\' # 'D:\\project\\bottlecap\\test1\\out\\0717\\'
# img_files = os.listdir(img_path)
# img_path = 'e:\\logo\\red\\2024-10-30\\2\\resultNG\\'

# color_t = 'red'

color_t, img_path = read_path_color()

work_path = 'temp_o/'+color_t
sens = 6
defor = 5

param = set_blob_param(color_t, 'Settings/oblob-param-newcam.xml') # add D:/project/bottlecap/code/
max_Area = param[3]
minSize = param[2] * (11 - sens) / 5
# rate = (11 - sens) / 5
print('MinSize = ', minSize, int(param[14] * (11-defor) / 5))
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
slopes_pt = []
corner_points = []

if not os.path.exists(work_path):
    os.makedirs(work_path)
    os.makedirs(work_path + '_p/')
img_files = [_ for _ in os.listdir(img_path) if (_.endswith('.jpg') or _.endswith('.png') or _.endswith('.bmp'))]
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
    # 241210 add "contours, bin_center, radius"
    img_bin, img_o7, img_o7_inv, center, contours, bin_center, radius = process_img(frame_res, mask, frame.shape[1], frame.shape[0])

    #極座標處理
    polar_img = cv.linearPolar(img_o7_inv, (img_o7.shape[1]//2, img_o7.shape[0]//2), img_o7.shape[1]//2+10, cv.WARP_FILL_OUTLIERS)
    edges =cv.Canny(polar_img, 50, 150)
    #cv.imwrite('temp_o/'+color_t+ 'polar'+ img_file + '_b.png', polar_img)
    #cv.imwrite('temp_o/'+color_t+ 'edges'+ img_file + '_b.png', edges)

    # edges_process(frame_res, img_o7_inv)

    cv.imwrite('temp_o/'+color_t+ '_p/'+ img_file + '_b.png', img_bin)
    cv.imwrite('temp_o/' + color_t + '_p/' + img_file + '_o7.png', img_o7)
    cv.imwrite('temp_o/'+color_t+ '_p/'+ img_file + '_o7_inv.png', img_o7_inv)
    # 241210 add "contours, bin_center, radius"
    img_res = process_blob(frame, frame_res, img_bin, img_o7_inv, contours, bin_center, radius)
#-----------------------------------------------------------------------
    #240822 contour_img, corner_points, slopes_pt = edges_process(frame, img_o7_inv)
    if param[15] == 1:
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