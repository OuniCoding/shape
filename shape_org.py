# findContours(image, mode, method, contours=None, hierarchy=None, offset=None)
# - image輸入/輸出的二值圖像
# - mode 迒回輪廓的結構、可以是List、Tree、External
# - method 輪廓點的編碼方式，基本是基於鏈式編碼
# - contours 迒回的輪廓集合
# - hierarchy 迒回的輪廓層次關系
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
import time

class ShapeAnalysis:
    def __init__(self):
        self.shapes = {'triangle': 0, 'rectangle': 0, 'polygons': 0, 'circles': 0}

    def analysis(self, frame):
        h, w, ch = frame.shape
        #frame = cv.resize(frame, (int(w*0.8), int(h*0.8)), interpolation=cv.INTER_LINEAR)
        # h, w, ch = frame.shape
        result = np.zeros((h, w, ch), dtype=np.uint8)
        # 二值化圖像
        print("start to detect lines...\n")
        begin_time = time.time()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # img blur
        # gray2 = cv.medianBlur(gray, 7)
        gray2 = cv.GaussianBlur(gray, (5, 5), 0)
        # ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
        #白:64, 透明:80, gold:70, red:91 ; New=88
        # red1/blue1/green1=82; pk1/lblue1/gold1/trans1=88; whitle1=98; black=?
        ret, binary = cv.threshold(gray2, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # ret, binary = cv.threshold(gray2, 88, 255, cv.THRESH_BINARY)    # | cv.THRESH_OTSU)
        output2 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
        output3 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        output4 = cv.adaptiveThreshold(gray2, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 13, 2)
        output5 = cv.adaptiveThreshold(gray2, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 13, 2)
        cv.imshow("input image", frame)
        cv.imshow("Gray", gray)
        cv.imshow("Binary", binary)
        cv.imshow("O2", output2)
        cv.imshow("O3", output3)
        cv.imshow("O4", output4)
        cv.imshow("O5", output5)
        # cv2.addWeighted(img1, alpha, img2, beta, gamma)
        # 計算公式：img1*alpha + img2*beta + gamma
        output6 = cv.addWeighted(gray, 0.3, output5, 0.5, 50)
        cv.imshow("O6", output6)
        output7 = cv.subtract(output5, gray)
        cv.imshow("O7", output7)

        cv.waitKey(0)

        #out_binary, contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        amount = 0
        for cnt in range(len(contours)):
            # 提取與繪制輪廓
            cv.drawContours(result, contours, cnt, (0, 255, 0), 1)

            # 輪廓逼近
            epsilon = 0.01 * cv.arcLength(contours[cnt], True)
            approx = cv.approxPolyDP(contours[cnt], epsilon, True)

            # print("approx=",approx)     # 輪廓邊線座標

            # 分析幾何形狀
            corners = len(approx)
            # looking bottom coordinate
            y_max = 0
            x_max = 0
            y = 0
            x = 0
            for c in range(corners):
                [xy] = approx[c]
                y = xy[1:2]
                x = xy[:1]
                if y > y_max :
                    y_max = int(y)
                    x_max = int(x)
                    [xy_previous] = approx[c-1]

            shape_type = ""
            if corners == 3:
                count = self.shapes['triangle']
                count = count+1
                self.shapes['triangle'] = count
                shape_type = "三角形"
            if corners == 4:
                count = self.shapes['rectangle']
                count = count + 1
                self.shapes['rectangle'] = count
                shape_type = "矩形"
            if corners >= 10:
                count = self.shapes['circles']
                count = count + 1
                self.shapes['circles'] = count
                shape_type = "圓形"
            if 4 < corners < 10:
                count = self.shapes['polygons']
                count = count + 1
                self.shapes['polygons'] = count
                shape_type = "多邊形"

            # 求解中心位置
            mm = cv.moments(contours[cnt])
            # print(corners,shape_type, mm)
            try:
                if shape_type != "":
                    cx = int(mm['m10'] / mm['m00'])
                    cy = int(mm['m01'] / mm['m00'])
                    cv.circle(result, (cx, cy), 1, (0, 0, 255), -1)

                    amount = amount + 1
                    # 顏色分析
                    color = frame[cy][cx]
                    color_str = "(" + str(color[0]) + ", " + str(color[1]) + ", " + str(color[2]) + ")"
                    cv.putText(result, str(amount), (cx+5, cy+5), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1)

                    # 計算面積與周長
                    p = cv.arcLength(contours[cnt], True)
                    area = cv.contourArea(contours[cnt])

                    y = xy_previous[1]
                    x = xy_previous[0]
                    dis = ((x-x_max)**2+(y-y_max)**2)**0.5
                    print(xy_previous)

                    #print("%s 中心座標: %s 周長: %.3f, 面積: %.3f 顏色: %s 形狀: %s (%2i) 底部座標(%3i, %3i)" % (amount, (cx,cy), p, area, color_str, shape_type, corners, x_max, y_max))
                    print("%s 中心座標: %s 顏色: %s 形狀: %s (%2i) 底部座標(%3i, %3i) 距離: %3.3f" % (amount, (cx, cy), color_str, shape_type, corners, x_max, y_max, dis))

                    # mark bottom coordinate
                    #cv.circle(result, (x_max, y_max), 1, (255, 255, 255), -1)
            except:
                continue

        print("執行時間：%f 秒" % (time.time() - begin_time))
        cv.imshow("Analysis Result", self.draw_text_info(result))
        cv.imwrite("test-result.png", self.draw_text_info(result))
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
    #image = cv.imread('F:/project/bottlecap/trans/Image_20240131183812854.jpg')  #B3.jpg') # ("shapes2.png") #('a-test.png') #
    #image = cv.imread('F:/project/bottlecap/white/NG/Image_20240201174432182.jpg')
    #image = cv.imread('F:/project/bottlecap/white/OK/Image_20240201181806331.jpg')
    #image = cv.imread('F:/project/bottlecap/gold/Image_20240202184242330.jpg')
    #image = cv.imread('F:/project/bottlecap/red/OK/Image_20240202154544577.jpg')   #Image_20240202160555332.jpg')
    #image = cv.imread('F:/project/bottlecap/20240217 outside/trans/Image_20240217134201707.jpg')    #Image_20240217133233378.jpg') 180
    image = cv.imread('F:\\project\\bottlecap\\test\\Image_20240229171004566.jpg ')
    ld = ShapeAnalysis()
    ld.analysis(image)

    cv.waitKey(0)
    cv.destroyAllWindows()