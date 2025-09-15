import cv2
import numpy as np
import time

#img = cv2.imread('b.png', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('B3.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
se = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, se)

params = cv2.SimpleBlobDetector_Params()
detector = cv2.SimpleBlobDetector.create(params)

keypoints = detector.detect(binary)

img_with_keypoints = cv2.drawKeypoints(binary, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
blob_info = []
for kp in keypoints:
    cv2.circle(img, (np.int(kp.pt[0]), np.int(kp.pt[1])), 3, (0, 255, 0), -1, 8)
    cv2.circle(img, (np.int(kp.pt[0]), np.int(kp.pt[1])), np.int(kp.size/2), (0, 0, 255), 2, 8)

cv2.imshow('Binary', binary)
cv2.imshow('BLOB', img_with_keypoints)
cv2.imshow('Keypoints', img)
cv2.waitKey(0)
cv2.destroyAllWindows()