#!/usr/bin/python

import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import numpy as np
import csv

colours = [(230, 63, 7), (48, 18, 59), (68, 81, 191), (69, 138, 252), (37, 192, 231), (31, 233, 175), (101, 253, 105), (175, 250, 55), (227, 219, 56), (253, 172, 52), (246, 108, 25), (216, 55, 6), (164, 19, 1), (90, 66, 98), (105, 116, 203), (106, 161, 253), (81, 205, 236), (76, 237, 191), (132, 253, 135), (191, 251, 95), (233, 226, 96), (254, 189, 93), (248, 137, 71), (224, 95, 56), (182, 66, 52), (230, 63, 7), (48, 18, 59), (68, 81, 191), (69, 138, 252), (37, 192, 231), (31, 233, 175), (101, 253, 105), (175, 250, 55), (227, 219, 56), (253, 172, 52), (246, 108, 25), (216, 55, 6), (164, 19, 1), (90, 66, 98), (105, 116, 203), (106, 161, 253), (81, 205, 236), (76, 237, 191), (132, 253, 135), (191, 251, 95), (233, 226, 96), (254, 189, 93), (248, 137, 71), (224, 95, 56), (182, 66, 52)]

img = cv2.imread('F:\\project\\bottlecap\\test\\Image_20240229171004566.jpg') #Read image australian_states.png
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to greyscale
grey_inv = cv2.bitwise_not(gray) #Inverse
cv2.imshow('G', grey_inv)
gray2 = cv2.GaussianBlur(gray, (5, 5), 0)
output5 = cv2.adaptiveThreshold(gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2)
output7 = cv2.subtract(output5, gray)
cv2.imshow("O7", output7)

cv2.waitKey(0)
ret,thresh_au = cv2.threshold(output7,128,255,cv2.THRESH_BINARY) #Threshold
cv2.imshow('Gray', thresh_au)
cv2.waitKey(0)
def blob_properties(contours):
  cont_props= []
  i = 0
  for cnt in contours:
    area= cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt,True)
    convexity = cv2.isContourConvex(cnt)
    x1,y1,w,h = cv2.boundingRect(cnt)
    x2 = x1+w
    y2 = y1+h
    aspect_ratio = float(w)/h
    rect_area = w*h
    extent = float(area)/rect_area
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area
    # (xa,ya),(MA,ma),angle = cv2.fitEllipse(cnt)
    rect = cv2.minAreaRect(cnt)
    (xc,yc),radius = cv2.minEnclosingCircle(cnt)
    # ellipse = cv2.fitEllipse(cnt)
    rows,cols = img.shape[:2]
    [vx,vy,xf,yf] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
    lefty = int((-xf*vy/vx) + yf)
    righty = int(((cols-xf)*vy/vx)+yf)
    # Add parameters to list
    # add = i+1, area, round(perimeter, 1), convexity, round(aspect_ratio, 3), round(extent, 3), w, h, round(hull_area, 1), round(angle, 1), x1, y1, x2, y2, round(radius, 6), xa, ya, xc, yc, xf[0], yf[0], rect, ellipse, vx[0], vy[0], lefty, righty
    #       0    1     2                    3          4                       5                 6  7  8                    9                10  11  12  13  14                15  16  17  18  19     20     21    22       23     24     25     26
    add = i+1, area, round(perimeter, 1), convexity, round(aspect_ratio, 3), round(extent, 3), w, h, round(hull_area, 1), x1, y1, x2, y2, round(radius, 6), xc, yc, xf[0], yf[0], rect, vx[0], vy[0], lefty, righty
    #       0    1   2                    3          4                       5                 6  7  8                    9   10  11  12  13                14  15  16     17     18    19     20     21     22
    cont_props.append(add)
    i += 1
  return cont_props

contours = cv2.findContours(thresh_au, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
# Sort contours
sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
blobs_data = blob_properties(sorted_contours)

# Plot contours
image_plot = img.copy()

i = 0
for rows in blobs_data:
    pos = blobs_data[i]
    inverted_colours = (255-colours[i][0],255-colours[i][1],255-colours[i][2])
    cv2.drawContours(image_plot, [sorted_contours[i]], -1, colours[i], -1) #, colours[i], thickness=cv2.FILLED)
    #cv2.rectangle(image_plot, (pos[10], pos[11]), (pos[12], pos[13]), inverted_colours, 2)
    cv2.putText(image_plot, str(pos[0]), (int(pos[19]), int(pos[20])), cv2.FONT_HERSHEY_SIMPLEX, 1, inverted_colours, 2, cv2.LINE_AA)
    i += 1
plt.imshow(image_plot)

# Save image
cv2.imwrite("temp/australian_states_blobs_ids.png", image_plot)

# Collect data
header = ['blob_id','area','perimeter','convexity','aspect_ratio','extent','width','height','hull_area','ellipse_angle','rect_x1','rect_y1','rect_x2','rect_y2','radius','xa','ya','xc','yc','xf','yf','min_area_rectangle','ellipse','vx','vy','left_y','right_y']
with open("temp/australian_states.csv", 'w', newline='') as file:
    dw = csv.DictWriter(file, delimiter=',', fieldnames=header)
    dw.writeheader()
    writer = csv.writer(file)
    writer.writerows(blobs_data)

# Grab stats to plot for biggest blob

# Rotated rectangle
# rect = blobs_data[0][21]
rect = blobs_data[0][18]
box = cv2.boxPoints(rect)
box = np.int0(box)

# Mimimum enclosing circle
# centre = (int(blobs_data[0][17]),int(blobs_data[0][18]))
# radius = int(blobs_data[0][14])
centre = (int(blobs_data[0][14]),int(blobs_data[0][15]))
radius = int(blobs_data[0][13])

# Plot the parameters on the biggest blob
image_plot2 = img.copy()
pos = blobs_data[0]
rows,cols = img.shape[:2]
# cv2.rectangle(image_plot2, (pos[10], pos[11]), (pos[12], pos[13]), colours[1], 2) # Bounding rectangle
cv2.rectangle(image_plot2, (pos[9], pos[10]), (pos[11], pos[12]), colours[1], 2) # Bounding rectangle
cv2.drawContours(image_plot2,[box],0,colours[4],2) # Rotated rectangle
cv2.circle(image_plot2,centre,radius,colours[7],2) # Minimum Enclosing Circle
# cv2.ellipse(image_plot2,pos[22],colours[10],2) # Fitted ellipse
# cv2.line(image_plot2,(cols-1,pos[26]),(0,pos[25]),colours[14],2) # Fitted line
cv2.line(image_plot2,(cols-1,pos[22]),(0,pos[21]),colours[14],2) # Fitted line
plt.imshow(image_plot2)

# Save image
cv2.imwrite("temp/australian_states_blobs_biggest.png", image_plot2)