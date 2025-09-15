# findContours(image, mode, method, contours=None, hierarchy=None, offset=None)
# - image輸入/輸出的二值圖像
# - mode 迒回輪廓的結構、可以是List、Tree、External
# - method 輪廓點的編碼方式，基本是基於鏈式編碼
# - contours 迒回的輪廓集合
# - hieracrchy 迒回的輪廓層次關系
# - offset 點是否有位移
#
# approxPolyDP(curve, epsilon, closed, approxCurve=None)
# - curve 表示輸入的輪廓點集合
# - epsilon 表示逼近曲率，越小表示相似逼近越厲害
# - close 是否閉合
#
# moments(array, binaryImage=None)
# - array表示指定輸入輪廓
# - binaryImage默認為None
####################################################
import cv2 as cv
import numpy as np
import math

class ShapeAnalysis:
    def __init__(self):
        self.shapes = {'triangle': 0, 'rectangle': 0, 'polygons': 0, 'circles': 0}

    def analysis(self, frame):
        h, w, ch = frame.shape
        result = np.zeros((h, w, ch), dtype=np.uint8)
        # 二值化圖像
        print("start to detect lines...\n")
        #hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        #gray = cv.cvtColor(hsv, cv.COLOR_BGR2GRAY)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # img blur
        blur = cv.GaussianBlur(gray, (3, 3), 0)
        ret, binary = cv.threshold(blur, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU) #50 40
        #cv.imshow("input image", frame)
        #cv.imshow("HSV", hsv)
        cv.imshow("Gray", gray)
        cv.imshow("Binary", binary)
        cv.waitKey(0)

        # out_binary, contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        amount = 0
        for cnt in range(len(contours)):
            # 提取與繪制輪廓
            # cv.drawContours(frame, contours, cnt, (0, 255, 0), 2)
            # cv.drawContours(result, contours, cnt, (0, 255, 0), 2)

            # 輪廓逼近
            epsilon = 0.01 * cv.arcLength(contours[cnt], True)
            approx = cv.approxPolyDP(contours[cnt], epsilon, True)

            # 計算面積與周長
            p = cv.arcLength(contours[cnt], True)
            r = p / 2 / math.pi
            area = cv.contourArea(contours[cnt])
            if r < 185:
                continue

            # 分析幾何形狀
            corners = len(approx)
            shape_type = ""
#            if corners == 3:
#                count = self.shapes['triangle']
#                count = count+1
#                self.shapes['triangle'] = count
#                shape_type = "三角形"
#            if corners == 4:
#                count = self.shapes['rectangle']
#                count = count + 1
#                self.shapes['rectangle'] = count
#                shape_type = "矩形"
            if corners >= 10:
                count = self.shapes['circles']
                count = count + 1
                self.shapes['circles'] = count
                shape_type = "圓形"
            if 4 < corners < 10:
                continue
                count = self.shapes['polygons']
                count = count + 1
                self.shapes['polygons'] = count
                shape_type = "多邊形"

            # 求解中心位置
            mm = cv.moments(contours[cnt])
            # print(corners,shape_type, mm)
            if shape_type != "":
                cx = int(mm['m10'] / mm['m00'])
                cy = int(mm['m01'] / mm['m00'])
                # 顏色分析
                color = frame[cy][cx]
                color_str = "(" + str(color[0]) + ", " + str(color[1]) + ", " + str(color[2]) + ")"

                # 重心點標示紅色
                # cv.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                # cv.circle(result, (cx, cy), 3, (0, 0, 255), -1)



                # 最小外接圓
                # (x,y), radius = cv.minEnclosingCircle(contours[cnt])
                # x = int(x)
                # y = int(y)
                # print(int(x), int(y), radius)
                # cv.line(frame, (x - 10, y), (x + 10, y), (0, 255, 255), thickness=1)
                # cv.line(frame, (x, y - 10), (x, y + 10), (0, 255, 255), thickness=1)
                # cv.circle(frame, (int(x), int(y)), int(radius), (255, 255, 0), 2)
                # 最大內接圓
                ellipse = cv.fitEllipse(contours[cnt]) # ellipse = [(中心座標),(橢圓長軸長度,橢圓短軸長度),橢圓旋轉角度]
                (x, y) = ellipse[0]
                (longRadius,shortRadius) = ellipse[1]
                x = int(x)
                y = int(y)
                radius = int(longRadius/2)
                print(int(x), int(y), radius)
                # cv.ellipse(frame, ellipse, (0, 255, 0), 2)
                cv.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                # 標示中心點
                cv.line(frame, (x - 10, y), (x + 10, y), (0, 0, 255), thickness=2)
                cv.line(frame, (x, y - 10), (x, y + 10), (0, 0, 255), thickness=2)


                amount = amount + 1
                # 顏色分析
                # color = frame[cy][cx]
                # color_str = "(" + str(color[0]) + ", " + str(color[1]) + ", " + str(color[2]) + ")"
                cv.putText(frame, str(amount), (cx+5, cy+5), cv.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
                # cv.putText(result, str(amount), (cx + 5, cy + 5), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1)

                # 計算面積與周長
                # p = cv.arcLength(contours[cnt], True)
                # r = p/2/math.pi
                # area = cv.contourArea(contours[cnt])
                print("%s 中心座標: %s 半徑: %.3f, 周長: %.3f, 面積: %.3f 顏色: %s 形狀: %s (%2i)" % (amount, (cx,cy), r, p, area, color_str, shape_type, corners))

        cv.putText(frame, "circles: " + str(self.shapes['circles']), (10, 25), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 1)
        cv.imshow("Analysis Result", frame)
        cv.imwrite("test-result.png", frame)
        # cv.imshow("Analysis Result", self.draw_text_info(result))
        # cv.imwrite("test-result.png", self.draw_text_info(result))
        return self.shapes

    def draw_text_info(self, image):
        c1 = self.shapes['triangle']
        c2 = self.shapes['rectangle']
        c3 = self.shapes['polygons']
        c4 = self.shapes['circles']
        cv.putText(image, "triangle: " + str(c1), (10, 20), cv.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1)
        cv.putText(image, "rectangle: " + str(c2), (10, 40), cv.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1)
        cv.putText(image, "polygons: " + str(c3), (10, 60), cv.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1)
        cv.putText(image, "circles: " + str(c4), (10, 80), cv.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1)
        return image

if __name__ == "__main__":
    image_org = cv.imread('p1.jpg')  # ('cap/1red-5050.jpg') # ('447.bmp') #("result64.jpg") #('a-test.png') #
    w, h, s = image_org.shape
    image = cv.resize(image_org, (int(w/2), int(h/2)))

    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # 青&藍 (h:色調 s:飽和度 v:亮度)
    hmin = 78  # 78
    hmax = 90  # 124
    smin = 43
    smax = 255
    vmin = 46
    vmax = 255
    # 白
    hmin = 0   #hmin = 0
    hmax = 180  #hmax = 140
    smin = 0    #smin = 0
    smax = 32   #smax = 35
    vmin = 66   #vmin = 200
    vmax = 150  #vmax = 255
    # pink
    hmin = 0  # 5
    hmax = 167  # 180
    smin = 0  # 43
    smax = 255  # 215  #255
    vmin = 26  # 46
    vmax = 255  # 255

    hsv_low = np.array([hmin, smin, vmin])
    hsv_high = np.array([hmax, smax, vmax])
    mask = cv.inRange(hsv, hsv_low, hsv_high)
    res = cv.bitwise_and(image, image, mask=mask)
    res_old = res
    mask_old = mask
    cv.imshow('Input', image)
    cv.imshow('Mask_Result', res)
    # 印出該色的數量
    #print(cv.countNonZero(mask))
    amount = cv.countNonZero(mask)
    img_xor = cv.bitwise_xor(mask, mask_old)
    amount_r = cv.countNonZero(img_xor)
    print(amount_r)
#    if amount_r > 60:  # 自製影片雜訊問題
    ld = ShapeAnalysis()
    # ld.analysis(res)
    ld.analysis(image)

#    ld = ShapeAnalysis()
#    ld.analysis(src)
    cv.waitKey(0)
    cv.destroyAllWindows()