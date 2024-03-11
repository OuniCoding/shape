import cv2 as cv
import numpy as np
import os
# 黑          # 灰            # 白         # 紅                # 橙           # 黃          # 綠            # 青           # 藍          # 紫
# hmin = 0    # hmin = 0    # hmin = 0   # hmin = 0    156  # hmin = 11   # hmin = 26   # hmin = 35   # hmin = 78   # hmin = 100  # hmin = 125
# hmax = 180  # hmax = 180  # hmax = 18  # hmax = 10   180  # hmax = 25   # hmax = 34   # hmax = 77   # hmax = 99   # hmax = 124  # hmax = 155
# smin = 0    # smin = 0    # smin = 0   # smin = 43        # smin = 43   # smin = 43   # smin = 43   # smin = 43   # smin = 43   # smin = 43
# smax = 255  # smax = 43   # smax = 30  # smax = 255       # smax = 255  # smax = 255  # smax = 255  # smax = 255  # smax = 255  # smax = 255
# vmin = 0    # vmin = 46   # vmin = 22  # vmin = 46        # vmin = 46   # vmin = 46   # vmin = 46   # vmin = 46   # vmin = 46   # vmin = 46
# vmax = 46   # vmax = 220  # vmax = 25  # vmax = 255       # vmax = 255  # vmax = 255  # vmax = 255  # vmax = 255  # vmax = 255  # vmax = 255

image = cv.imread('result64.jpg') #result64.jpg00002.png') #("shapes.png") # ('a1.png')
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

# 白
hmin = 0
hmax = 140
smin = 0
smax = 35
vmin = 200
vmax = 255
hsv_low = np.array([hmin, smin, vmin])
hsv_high = np.array([hmax, smax, vmax])
mask = cv.inRange(hsv, hsv_low, hsv_high)
res = cv.bitwise_and(image, image, mask=mask)
# 印出該色的數量
print(cv.countNonZero(mask))
cv.imshow('Input', image)
cv.imshow('White_Result', res)
cv.imwrite('a-test.png', res)
cv.waitKey(0)
# 綠
hmin = 35
hmax = 77
smin = 43
smax = 255
vmin = 46
vmax = 255
hsv_low = np.array([hmin, smin, vmin])
hsv_high = np.array([hmax, smax, vmax])
mask = cv.inRange(hsv, hsv_low, hsv_high)
res = cv.bitwise_and(image, image, mask=mask)
# 印出該色的數量
print(cv.countNonZero(mask))
cv.imshow('Green_Result', res)
cv.waitKey(0)
# 青&藍
hmin = 78 #78
hmax = 90 #124
smin = 43
smax = 255
vmin = 46
vmax = 255
hsv_low = np.array([hmin, smin, vmin])
hsv_high = np.array([hmax, smax, vmax])
mask = cv.inRange(hsv, hsv_low, hsv_high)
res = cv.bitwise_and(image, image, mask=mask)
# 印出該色的數量
print(cv.countNonZero(mask))
cv.imshow('Blue_Result', res)
cv.waitKey(0)
# 黃
hmin = 18
hmax = 34
smin = 43
smax = 255
vmin = 46
vmax = 255
hsv_low = np.array([hmin, smin, vmin])
hsv_high = np.array([hmax, smax, vmax])
mask = cv.inRange(hsv, hsv_low, hsv_high)
res = cv.bitwise_and(image, image, mask=mask)
# 印出該色的數量
print(cv.countNonZero(mask))
cv.imshow('Yellow_Result', res)
cv.waitKey(0)
# 紅
hmin = 0      #156
hmax = 10     #180
smin = 43
smax = 255
vmin = 46
vmax = 255
hsv_low = np.array([hmin, smin, vmin])
hsv_high = np.array([hmax, smax, vmax])
mask = cv.inRange(hsv, hsv_low, hsv_high)
res = cv.bitwise_and(image, image, mask=mask)
# 印出該色的數量
print(cv.countNonZero(mask))
cv.imshow('Red_Result', res)
cv.waitKey(0)
cv.destroyAllWindows()