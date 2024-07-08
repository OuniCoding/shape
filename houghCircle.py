import cv2
import numpy as np

image = cv2.imread('F:\\project\\bottlecap\\test1\\in\\white\\2024-07-01\\1\\20240701_13-31-44_965jpg_b.png')

gray =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray,(5, 5), 0)
cv2.imshow('blur', blurred)
edges = cv2.Canny(blurred, 50, 50)
cv2.imshow('canny', edges)
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 100, param1=50, param2=30, minRadius=0,maxRadius=0)

if circles is not None:
    circles = np.round(circles[0, :]).astype('int')

    for (x,y,r) in circles:
        cv2.circle(image, (x,y), r, (0,255,0), 2)

cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()