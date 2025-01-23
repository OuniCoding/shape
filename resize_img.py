#!/usr/bin/python

import cv2
import numpy as np
import os
import glob

color_t = 'Zsamples'
work_path = 'F:\\project\\bottlecap\\test\\'+color_t+'_416\\'
if not os.path.exists(work_path):
    os.makedirs(work_path)

img_path = 'F:\\project\\python\\shape\\temp\\' + color_t + '\\'  # 'F:\\project\\bottlecap\\Samples\\' + color_t + "\\"
# img_files = os.listdir(img_path)
img_files = [_ for _ in os.listdir(img_path) if _.endswith('.jpg')]
# types = os.path.join(img_path, '*.jpg'), os.path.join(img_path, '*.jpeg'), os.path.join(img_path, '*.png')
# files_grabbed = []
# for files in types:
#     files_grabbed.extend(sorted(glob.iglob(files)))

for img_file in img_files:  # files_grabbed:
    # frame = cv2.imread(img_file)
    frame = cv2.imread(img_path + img_file)
    res_img = cv2.resize(frame,(416, 416))
    cv2.imwrite(work_path + '416_'+ img_file , res_img)

cv2.destroyAllWindows()
print('Resize images, work path:', work_path)