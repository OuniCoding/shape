import cv2 as cv
import numpy as np
import time

class ShapeAnalysis:
    def __init__(self):
        self.shapes = {'triangle': 0, 'rectangle': 0, 'polygons': 0, 'circles': 0}

    def analysis(self, frame):
        h, w, ch = frame.shape
        result = np.zeros((h, w, ch), dtype=np.uint8)
        # 二值化圖像
        #print("start to detect lines...\n")
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        gray = cv.cvtColor(hsv, cv.COLOR_BGR2GRAY)
        # imgBlur = cv.GaussianBlur(gray, (5, 5), 0)
        imgBlur = cv.blur(gray, (5, 5))
        ret, binary = cv.threshold(imgBlur, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        #cv.imshow("input image", frame)
        #cv.imshow("HSV", hsv)
        #cv.imshow("Gray", gray)
        cv.imshow("Binary", binary)
        #cv.waitKey(0)

        # out_binary, contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        tag_amount = 0
        for cnt in range(len(contours)):
            # 提取與繪制輪廓
            cv.drawContours(result, contours, cnt, (0, 255, 0), 2)

            # 輪廓逼近
            epsilon = 0.01 * cv.arcLength(contours[cnt], True)
            approx = cv.approxPolyDP(contours[cnt], epsilon, True)

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
 #           if 4 < corners < 10:
 #               count = self.shapes['polygons']
 #               count = count + 1
 #               self.shapes['polygons'] = count
 #               shape_type = "多邊形"

            # 求解中心位置
            mm = cv.moments(contours[cnt])
            # print(corners,shape_type, mm)
            if shape_type != "":
                cx = int(mm['m10'] / mm['m00'])
                cy = int(mm['m01'] / mm['m00'])
                cv.circle(result, (cx, cy), 3, (0, 0, 255), -1)

                tag_amount = tag_amount + 1
                # 顏色分析
                color = frame[cy][cx]
                color_str = "(" + str(color[0]) + ", " + str(color[1]) + ", " + str(color[2]) + ")"
                cv.putText(result, str(tag_amount), (cx+5, cy+5), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1)

                # 計算面積與周長
                p = cv.arcLength(contours[cnt], True)
                area = cv.contourArea(contours[cnt])
                print(" 中心座標: %s 周長: %.3f, 面積: %.3f 顏色: %s 形狀: %s " % ((cx,cy), p, area, color_str, shape_type))

        cv.imshow("Analysis Result", self.draw_text_info(result))
        return self.shapes

    def draw_text_info(self, image):
        # c1 = self.shapes['triangle']
        # c2 = self.shapes['rectangle']
        # c3 = self.shapes['polygons']
        c4 = self.shapes['circles']
        # cv.putText(image, "triangle: " + str(c1), (10, 80), cv.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1)
        # cv.putText(image, "rectangle: " + str(c2), (10, 40), cv.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1)
        # cv.putText(image, "polygons: " + str(c3), (10, 60), cv.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1)
        cv.putText(image, "circles: " + str(c4), (10, 20), cv.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1)
        return image

# 黑          # 灰            # 白         # 紅                # 橙           # 黃          # 綠            # 青           # 藍          # 紫
# hmin = 0    # hmin = 0    # hmin = 0   # hmin = 0    156  # hmin = 11   # hmin = 26   # hmin = 35   # hmin = 78   # hmin = 100  # hmin = 125
# hmax = 180  # hmax = 180  # hmax = 18  # hmax = 10   180  # hmax = 25   # hmax = 34   # hmax = 77   # hmax = 99   # hmax = 124  # hmax = 155
# smin = 0    # smin = 0    # smin = 0   # smin = 43        # smin = 43   # smin = 43   # smin = 43   # smin = 43   # smin = 43   # smin = 43
# smax = 255  # smax = 43   # smax = 30  # smax = 255       # smax = 255  # smax = 255  # smax = 255  # smax = 255  # smax = 255  # smax = 255
# vmin = 0    # vmin = 46   # vmin = 22  # vmin = 46        # vmin = 46   # vmin = 46   # vmin = 46   # vmin = 46   # vmin = 46   # vmin = 46
# vmax = 46   # vmax = 220  # vmax = 25  # vmax = 255       # vmax = 255  # vmax = 255  # vmax = 255  # vmax = 255  # vmax = 255  # vmax = 255

begin_time = time.time()
start_time = begin_time
#image = cv.imread('a0.png')
#hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

cap = cv.VideoCapture(0) #"T1.mp4") #('o_60fps.mp4')#
w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))
print("w=",w,"h=",h,"fps=",fps)

#color = 0 # 灰階; 1:彩色
#fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
#out = cv2.VideoWriter("output.mp4", fourcc, fps, (w, h), color)

#計數檢測不同frame數量
i = 0
f = 0
c = 0
amount = 0

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        hmin = 0
        hmax = 140
        smin = 0
        smax = 35
        vmin = 200
        vmax = 255

        hsv_low = np.array([hmin, smin, vmin])
        hsv_high = np.array([hmax, smax, vmax])
        mask = cv.inRange(hsv, hsv_low, hsv_high)
        res = cv.bitwise_and(frame, frame, mask=mask)

        if f <= 2:
            res_old = res
            mask_old = mask

        # 印出該色的數量
        amount = cv.countNonZero(mask)
        img_xor = cv.bitwise_xor(mask, mask_old)
        amount_r = cv.countNonZero(img_xor)
        # print(f, "-", amount, " ", amount_r)
        if  amount_r > 0 : #自製影片雜訊問題
            i = i + 1
            print("Counter=",i," Frame=",f," ",amount_r," ",amount)
#            cv.imshow('video %s'%i, res)
#            cv.imshow('img %s' % f, img_xor)
            ld = ShapeAnalysis()
            ld.analysis(res)
        f += 1
        c += 1

        if (time.time() - start_time) != 0:  # 实时显示帧数
            cv.putText(frame, "FPS {0}".format(float('%.1f' % (c / (time.time() - start_time)))), (500, 50),
                        cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
                        3)
            # src = cv.resize(frame, (w // 2, h // 2), interpolation=cv.INTER_CUBIC)  # 窗口大小
            src = frame
            cv.imshow("image", src)
            print("FPS: ", c / (time.time() - start_time))
            c = 0
            start_time = time.time()
        time.sleep(0.8 / fps)  # 按原帧率播放


    else:
        break
    k = cv.waitKey(1)
    if k == 27:
        break


end_time = time.time()
print("Frames = %s, 執行時間：%f 秒" % (f, (end_time - begin_time)))
print("press any key to stop!!")
cv.waitKey(0)
cap.release()
cv.destroyAllWindows()