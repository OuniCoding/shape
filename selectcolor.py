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

#image = cv.imread('result64.jpg') #result64.jpg00002.png') #("shapes.png") # ('a1.png')
#image = cv.imread('./temp/blue_416/416_Image_20240312084428698.jpg')
image = cv.imread('F:\\project\\bottlecap\\Samples\\black\\nImage_20240318160450314.jpg') #nImage_20240326113746271.jpg') #
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

# 白
hmin = 0
hmax = 180  #140
smin = 0
smax = 66   #53   #35
vmin = 22   #200
vmax = 255  #220
hsv_low = np.array([hmin, smin, vmin])
hsv_high = np.array([hmax, smax, vmax])
mask = cv.inRange(hsv, hsv_low, hsv_high)
res = cv.bitwise_and(image, image, mask=mask)
# 印出該色的數量
print(cv.countNonZero(mask))
cv.imshow('Input', image)
cv.imshow('White_Result', res)
cv.waitKey(0)
# 黑
hmin = 0
hmax = 48  #140
smin = 0
smax = 255
vmin = 0
vmax = 90   #46
hsv_low = np.array([hmin, smin, vmin])
hsv_high = np.array([hmax, smax, vmax])
mask = cv.inRange(hsv, hsv_low, hsv_high)
res = cv.bitwise_and(image, image, mask=mask)
# 印出該色的數量
print(cv.countNonZero(mask))
cv.imshow('Black_Result', res)
cv.waitKey(0)
# 透明
hmin = 0
hmax = 180  #140
smin = 0
smax = 68
vmin = 22
vmax = 255
hsv_low = np.array([hmin, smin, vmin])
hsv_high = np.array([hmax, smax, vmax])
mask = cv.inRange(hsv, hsv_low, hsv_high)
res = cv.bitwise_and(image, image, mask=mask)
# 印出該色的數量
print(cv.countNonZero(mask))
cv.imshow('Transparent_Result', res)
cv.waitKey(0)
# 綠
hmin = 35   #35
hmax = 164  #77
smin = 0    #43
smax = 220  #255
vmin = 46   #46
vmax = 255  #255
hsv_low = np.array([hmin, smin, vmin])
hsv_high = np.array([hmax, smax, vmax])
mask = cv.inRange(hsv, hsv_low, hsv_high)
res = cv.bitwise_and(image, image, mask=mask)
# 印出該色的數量
print(cv.countNonZero(mask))
cv.imshow('Green_Result', res)
cv.waitKey(0)
# 青&藍
hmin = 44
hmax = 164  #108
smin = 0
smax = 255  #255
vmin = 0   #46
vmax = 255  #255
hsv_low = np.array([hmin, smin, vmin])
hsv_high = np.array([hmax, smax, vmax])
mask = cv.inRange(hsv, hsv_low, hsv_high)
res = cv.bitwise_and(image, image, mask=mask)
# 印出該色的數量
print(cv.countNonZero(mask))
cv.imshow('Lightblue_Result', res)
cv.waitKey(0)
# 藍
hmin = 25 # hmin = 100  #109
hmax = 160   # hmax = 144  #124
smin = 0   # smin = 43
smax = 255  # smax = 255
vmin = 0   # vmin = 46
vmax = 255  # vmax = 255
hsv_low = np.array([hmin, smin, vmin])
hsv_high = np.array([hmax, smax, vmax])
mask = cv.inRange(hsv, hsv_low, hsv_high)
res = cv.bitwise_and(image, image, mask=mask)
# 印出該色的數量
print(cv.countNonZero(mask))
cv.imshow('Blue_Result', res)
cv.waitKey(0)
# 紅
hmin = 0      #156
hmax = 35     #180
smin = 0   #43
smax = 245  #255
vmin = 0   #46
vmax = 255  #255
hsv_low = np.array([hmin, smin, vmin])
hsv_high = np.array([hmax, smax, vmax])
mask = cv.inRange(hsv, hsv_low, hsv_high)
res1 = cv.bitwise_and(image, image, mask=mask)
# 印出該色的數量
print(cv.countNonZero(mask))
hmin = 80      #156
hmax = 255     #180
smin = 0   #43
smax = 255  #255
vmin = 46   #46
vmax = 255  #255
hsv_low = np.array([hmin, smin, vmin])
hsv_high = np.array([hmax, smax, vmax])
mask = cv.inRange(hsv, hsv_low, hsv_high)
res2 = cv.bitwise_and(image, image, mask=mask)
# 印出該色的數量
print(cv.countNonZero(mask))
res = cv.bitwise_or(res1, res2)
cv.imshow('Red_Result', res)
cv.waitKey(0)
# 金
hmin = 5       #156
hmax = 178  #132     #180
smin = 0    #15   #43
smax = 247  #255
vmin = 0   #46
vmax = 255  #255
hsv_low = np.array([hmin, smin, vmin])
hsv_high = np.array([hmax, smax, vmax])
mask = cv.inRange(hsv, hsv_low, hsv_high)
res = cv.bitwise_and(image, image, mask=mask)
# 印出該色的數量
print(cv.countNonZero(mask))
cv.imshow('Gold_Result', res)
cv.waitKey(0)
# 粉紅
hmin = 0  #5
hmax = 178  #180
smin = 0   #43
smax = 255  #215  #255
vmin = 10   #46
vmax = 255  #255
hsv_low = np.array([hmin, smin, vmin])
hsv_high = np.array([hmax, smax, vmax])
mask = cv.inRange(hsv, hsv_low, hsv_high)
res = cv.bitwise_and(image, image, mask=mask)
# 印出該色的數量
print(cv.countNonZero(mask))
cv.imshow('pink_Result', res)
cv.waitKey(0)

cv.destroyAllWindows()