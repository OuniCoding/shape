import cv2
import os
import numpy as np
import xml.etree.ElementTree as ET

def set_blob_param(category,para_name):
    param_file = ET.parse(para_name)
    root = param_file.getroot()
    id = 0
    param = []
    while root[0][id].tag != category:
        id += 1
    for i in range(0,13):
        param.append(int(root[0][id][i].attrib['value']))

    return param
#image = cv2.imread('F:\\project\\bottlecap\\test1\\Image_20240621113923008.jpg')
img_path = 'F:\\project\\bottlecap\\test1\\in\\blue\\2024-07-10\\1\\resultNG\\'

img_files = [_ for _ in os.listdir(img_path) if (_.endswith('.jpg') or _.endswith('.png'))]
img_file = img_files[0]
#image = cv2.imread('D:\\project\\bottlecap\\test1\\in\\green\\2024-07-10\\1\\resultG\\20240710_14-31-31_029.jpg')
image = cv2.imread(img_path+img_file)

cv2.imshow('BGR', image)
param = set_blob_param('blue', 'Settings/iblob-param.xml')
#white in
#hsv_low = np.array([0, 0, 98])
#hsv_high = np.array([221, 38, 255])
#white out
#hsv_low = np.array([0, 0, 54])
#hsv_high = np.array([221, 31, 220])

hsv_low = np.array([param[6], param[8], param[10]])
hsv_high = np.array([param[7], param[9], param[11]])

def h_low(value):
    hsv_low[0] = value
def h_high(value):
    hsv_high[0] = value
def s_low(value):
    hsv_low[1] = value
def s_high(value):
    hsv_high[1] = value
def v_low(value):
    hsv_low[2] = value
def v_high(value):
    hsv_high[2] = value

cv2.namedWindow('image')
cv2.createTrackbar('H low', 'image', 0, 255, h_low)
cv2.createTrackbar('H high', 'image', 0, 255, h_high)
cv2.createTrackbar('S low', 'image', 0, 255, s_low)
cv2.createTrackbar('S high', 'image', 0, 255, s_high)
cv2.createTrackbar('V low', 'image', 0, 255, v_low)
cv2.createTrackbar('V high', 'image', 0, 255, v_high)

cv2.setTrackbarPos('H low', 'image', hsv_low[0])
cv2.setTrackbarPos('S low', 'image', hsv_low[1])
cv2.setTrackbarPos('V low', 'image', hsv_low[2])
cv2.setTrackbarPos('H high', 'image', hsv_high[0])
cv2.setTrackbarPos('S high', 'image', hsv_high[1])
cv2.setTrackbarPos('V high', 'image', hsv_high[2])

index=0
while True:
    dst = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    dst = cv2.inRange(dst, hsv_low, hsv_high)
    cv2.imshow('dst', dst)
    key = cv2.waitKeyEx(1)
    if key & 0xff == ord('q'):
        break
    elif key == 2490368 or key == 2424832:
        index -= 1;
        if index < 0:
            index = len(img_files)-1
        img_file = img_files[index]
        # image = cv2.imread('D:\\project\\bottlecap\\test1\\in\\green\\2024-07-10\\1\\resultG\\20240710_14-31-31_029.jpg')
        image = cv2.imread(img_path+img_file)
        print(img_file)
    elif key == 2621440 or key == 2555904:
        index += 1;
        if index >= len(img_files):
            index = 0
        img_file = img_files[index]
        # image = cv2.imread('D:\\project\\bottlecap\\test1\\in\\green\\2024-07-10\\1\\resultG\\20240710_14-31-31_029.jpg')
        image = cv2.imread(img_path+img_file)
        cv2.imshow('BGR', image)
        print(img_file)

cv2.destroyAllWindows()

