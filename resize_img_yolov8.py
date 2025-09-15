#!/usr/bin/python

import cv2
import numpy as np
import os
import glob

color_t = r'\train'
work_path = r'E:\logo\white\OUTSIDE'+color_t+'_256'
work_path = r'E:\logo\L\2025-06-02\1\train_data1'
if not os.path.exists(work_path):
    os.makedirs(work_path)

img_path = r'E:\logo\white\OUTSIDE'+color_t
img_path = r'E:\logo\L\2025-06-02\1\train_data'
# 'F:\\project\\python\\shape\\temp\\' + color_t + '\\'  # 'F:\\project\\bottlecap\\Samples\\' + color_t + "\\"
# img_files = os.listdir(img_path)
img_files = [_ for _ in os.listdir(img_path) if (_.endswith('.jpg') or _.endswith('.bmp') or _.endswith('.png'))]
# types = os.path.join(img_path, '*.jpg'), os.path.join(img_path, '*.jpeg'), os.path.join(img_path, '*.png')
# files_grabbed = []
# for files in types:
#     files_grabbed.extend(sorted(glob.iglob(files)))

for img_file in img_files:  # files_grabbed:
    # frame = cv2.imread(img_file)
    frame = cv2.imread(img_path +'\\'+ img_file)
    if frame.shape[0] < 1024:
        res_img = cv2.resize(frame,(1024, 1024))
        # cv2.imwrite(work_path+'\\' + '256_'+ img_file , res_img)
        cv2.imwrite(work_path + '\\' + img_file, res_img)

cv2.destroyAllWindows()
print('Resize images, work path:', work_path)