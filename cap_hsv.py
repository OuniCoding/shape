import cv2
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
image = cv2.imread('F:\\project\\bottlecap\\test1\\in\\trans\\2024-07-10\\2\\resultG\\20240710_13-59-43_604.jpg')
#white 20240701_13-12-24_768.jpg  20240701_13-12-32_433.jpg  20240701_13-15-28_630.jpg
# 20240701_13-15-29_786.jpg  20240701_13-17-14_242.jpg  20240701_13-17-25_371.jpg  20240701_13-23-00_074.jpg
# 20240701_13-31-28_027.jpg  20240701_13-31-30_231.jpg  20240701_13-31-33_614.jpg
# 20240701_13-31-44_965.jpg  20240701_13-31-56_972.jpg  20240701_13-32-02_670.jpg
# 20240701_13-32-52_748.jpg  20240701_13-33-04_091.jpg  20240701_13-33-06_541.jpg
# 20240701_13-36-14_796.jpg  20240701_13-36-33_278.jpg  20240701_13-36-40_835.jpg
# 20240701_13-36-55_281.jpg  20240701_13-37-28_807.jpg  20240701_13-37-34_699.jpg
# 20240701_13-38-23_540.jpg

cv2.imshow('BGR', image)
param = set_blob_param('trans', 'Settings/oblob-param.xml')
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

while True:
    dst = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    dst = cv2.inRange(dst, hsv_low, hsv_high)
    cv2.imshow('dst', dst)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cv2.destroyAllWindows()

