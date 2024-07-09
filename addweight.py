import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import random

img = cv2.imread('D:/project/bottlecap/test1/in/white/2024-07-01/1/resultG/20240701_13-10-46_088.jpg')
bg = img.copy()
r = random.randint(0, 9)
match r:
    case 0:
        logo = cv2.imread('./point/point1120_1.jpg')
    case 1:
        logo = cv2.imread('./point/point1120_2.jpg')
    case 2:
        logo = cv2.imread('./point/point1120_3.jpg')
    case 3:
        logo = cv2.imread('./point/point1120_4.jpg')
    case 4:
        logo = cv2.imread('./point/point1120_5.jpg')
    case 5:
        logo = cv2.imread('./point/point1120_6.jpg')
    case 6:
        logo = cv2.imread('./point/point1120_7.jpg')
    case 7:
        logo = cv2.imread('./point/point1120_8.jpg')
    case 8:
        logo = cv2.imread('./point/point1120_9.jpg')
    case 9:
        logo = cv2.imread('./point/point1120_10.jpg')

#logo = cv2.imread('./point/point1120N.jpg')
output = cv2.addWeighted(img, 1, logo, 0.2, 1)

cv2.imshow('oxxostudio', output)
cv2.waitKey(0)      # 按下任意鍵停止


mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (0,0,1119,1119) #(100,1,421,378)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

cv2.imshow('o', img)
cv2.waitKey(0)      # 按下任意鍵停止

b_time = time.time()
#bg = img.copy()

size = logo.shape                                # 讀取 logo 的長寬尺寸

#img = np.zeros((1120,1120,3), dtype='uint8')       # 產生一張背景全黑的圖
img = np.zeros(size, dtype='uint8')       # 產生一張背景全黑的圖
#img[0:size[0], 0:size[1]] = '255'                        # 將圖片變成白色 ( 配合 logo 是白色底 )
img[0:size[0], 0:size[1]] = logo                 # 將圖片的指定區域，換成 logo 的圖案
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 產生一張灰階的圖片作為遮罩使用
ret, mask1  = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY_INV)  # 使用二值化的方法，產生黑白遮罩圖片
logo = cv2.bitwise_and(img, img, mask = mask1 )  # logo 套用遮罩

#bg = cv2.imread('meme.jpg')                      # 讀取底圖
ret, mask2  = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)      # 使用二值化的方法，產生黑白遮罩圖片
bg = cv2.bitwise_and(bg, bg, mask = mask2 )      # 底圖套用遮罩

output = cv2.add(bg, logo)                       # 使用 add 方法將底圖和 logo 合併
#cv2.putText(output, "NG", (1, nHeight - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
cv2.putText(output, "NG", (1, 1120 - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
cv2.imshow('studio', output)
e_time = time.time()
print(b_time, e_time, e_time- b_time)
cv2.imshow('min', cv2.resize(output, (int(1120*0.55), int(1120*0.55))))
cv2.waitKey(0)
cv2.destroyAllWindows()