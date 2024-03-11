#!/usr/bin/python

import cv2
import numpy as np
import os

color_t = 'trans'
work_path = 'temp/'+color_t+'_416/'
if not os.path.exists(work_path):
    os.makedirs(work_path)

img_path = 'F:\\project\\bottlecap\\test\\' + color_t + "\\"
img_files = os.listdir(img_path)
for img_file in img_files:
    frame = cv2.imread(img_path+img_file)
    res_img = cv2.resize(frame,(416, 416))
    cv2.imwrite(work_path + '416_'+ img_file , res_img)

cv2.destroyAllWindows()
print('Resize images, work path:', work_path)