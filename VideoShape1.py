"""
Project: Laser target recognition system
Created on Dec 26 21:47:00 2021
@author: Glan Tang
python3.8
install packsge: opencv 4.4.0.46
"""
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
        # print("start to detect lines...\n")
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        gray = cv.cvtColor(hsv, cv.COLOR_BGR2GRAY)
        # imgBlur = cv.GaussianBlur(gray, (5, 5), 0)
        imgBlur = cv.blur(gray, (3, 3))
        ret, binary = cv.threshold(imgBlur, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        # cv.imshow("input image", frame)
        # cv.imshow("HSV", hsv)
        # cv.imshow("Gray", gray)
        cv.imshow("Binary", binary)
        #cv.waitKey(0)

        contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        tag_amount = 0
        # test print('IN ',len(contours), contours)
        for cnt in range(len(contours)):
            # 提取與繪制輪廓
            cv.drawContours(result, contours, cnt, (0, 255, 0), 2)

            # 輪廓逼近
            epsilon = 0.01 * cv.arcLength(contours[cnt], True)
            approx = cv.approxPolyDP(contours[cnt], epsilon, True)

            # 分析幾何形狀
            corners = len(approx)
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
            if shape_type != "":
                cx = int(mm['m10'] / mm['m00'])
                cy = int(mm['m01'] / mm['m00'])
                cv.circle(result, (cx, cy), 3, (0, 0, 255), -1)

                tag_amount = tag_amount + 1
                # 顏色分析
                color = frame[cy][cx]
                color_str = "(" + str(color[0]) + ", " + str(color[1]) + ", " + str(color[2]) + ")"
                cv.putText(result, str(tag_amount), (cx+5, cy+5), cv.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 1)

                # 計算面積與周長
                p = cv.arcLength(contours[cnt], True)
                area = cv.contourArea(contours[cnt])
                # print(" 中心座標: %s 周長: %.3f, 面積: %.3f 顏色: %s 形狀: %s " % ((cx,cy), p, area, color_str, shape_type))
                print("* 中心座標: %s  形狀: %s " % ((cx, cy), shape_type))

        cv.imshow("Analysis Result", self.draw_text_info(result))
        return cx, cy   # center pos.    self.shapes

    def draw_text_info(self, image):
        c1 = self.shapes['triangle']
        c2 = self.shapes['rectangle']
        c3 = self.shapes['polygons']
        c4 = self.shapes['circles']
        cv.putText(image, "triangle: " + str(c1), (10, 70), cv.FONT_HERSHEY_PLAIN, 0.5, (255, 0, 0), 1)
        cv.putText(image, "rectangle: " + str(c2), (10, 50), cv.FONT_HERSHEY_PLAIN, 0.5, (255, 0, 0), 1)
        cv.putText(image, "polygons: " + str(c3), (10, 30), cv.FONT_HERSHEY_PLAIN, 0.5, (255, 0, 0), 1)
        cv.putText(image, "circles: " + str(c4), (10, 10), cv.FONT_HERSHEY_PLAIN, 0.5, (255, 0, 0), 1)
        return image

begin_time = time.time()
start_time = begin_time

cap = cv.VideoCapture("a1.mp4")
w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))
print("w=",w,"h=",h,"fps=",fps)

#計數檢測不同frame數量
i = 0
f = 0
c = 0
amount = 0

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        src = frame
        frame = cv.resize(frame, (w*2, h*2), interpolation=cv.INTER_AREA)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # imgBlur = cv.GaussianBlur(gray, (5, 5), 0)
        # ret, binary = cv.threshold(imgBlur, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

        if (time.time() - start_time) != 0:  # 实时显示帧数
            cv.putText(frame, "FPS {0}".format(float('%.1f' % (c / (time.time() - start_time)))), (5, 25),
                        cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                        1)
            # src = cv.resize(frame, (w // 2, h // 2), interpolation=cv.INTER_CUBIC)  # 窗口大小
            cv.imshow("Zoom Out", frame)
            # test print("FPS: ", c / (time.time() - start_time))
            c = 0
            start_time = time.time()

        if f == 0:
            res_old = gray
        diff = cv.subtract(gray, res_old)
        res = not np.any(diff)
        if res is False:
            cv.imshow('DIFFERENCE', diff)
            amount = cv.countNonZero(diff)
            # test print(amount)
            # test cv.waitKey(0)
            if amount > 0:
                diff = cv.GaussianBlur(diff, (5, 5), 0)
                diff = cv.cvtColor(diff, cv.COLOR_GRAY2BGR)
                # test cv.imshow('BGR', diff)

                hsv = cv.cvtColor(diff, cv.COLOR_BGR2HSV)
                hmin = 0
                hmax = 180
                smin = 0
                smax = 43
                vmin = 46
                vmax = 255

                hsv_low = np.array([hmin, smin, vmin])
                hsv_high = np.array([hmax, smax, vmax])
                mask = cv.inRange(hsv, hsv_low, hsv_high)
                res = cv.bitwise_and(diff, diff, mask=mask)
                cv.imshow('Detect', res)
                if f <= 2:
                    res_old = res
                    mask_old = mask

                # 印出該色的數量
                amount = cv.countNonZero(mask)
                img_xor = cv.bitwise_xor(mask, mask_old)
                amount_r = cv.countNonZero(img_xor)
                # test print(f, "-", amount, " ", amount_r)
                if amount_r > 0:  # 自製影片雜訊問題
                    i = i + 1
                    print("Counter=", i, " Frame=", f, " ", amount_r, " ", amount)
                    #            cv.imshow('video %s'%i, res)
                    #            cv.imshow('img %s' % f, img_xor)
                    res = cv.resize(res, (w, h), interpolation=cv.INTER_AREA)
                    ld = ShapeAnalysis()
                    px, py = ld.analysis(res)
                    # test print('POSTiON=', px, py)
                    # Drawing point
                    # cv.circle(src, (px, py), 1, (0, 0, 255), -1)
                    # Drawing cross
                    cv.line(src, (px-4, py), (px+4, py), (0, 0, 255), thickness=2)
                    cv.line(src, (px, py-4), (px, py+4), (0, 0, 255), thickness=2)
                    cv.imshow('Target',src)
                    print("   Press any key to continue!!", end=' ')
                    cv.waitKey(0)
                    print('Going.....')

        res_old = gray
        f += 1
        c += 1

        # time.sleep(1 / fps)  # 按原帧率播放
        # cv.waitKey(0)
    else:
        break
    k = cv.waitKey(1)
    if k == 27:
        break


end_time = time.time()
print("Frames = %s, 影片長度：%.3f 秒, 執行時間：%.3f 秒" % (f, (f/fps), (end_time - begin_time)))
print("   press any key to stop!!")
cv.waitKey(0)
cap.release()
cv.destroyAllWindows()