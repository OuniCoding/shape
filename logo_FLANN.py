#基於FLANN的匹配器(FLANN based Matcher)定位圖片
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

def img_rotate(src, angel):
    h = src.shape[0]
    w = src.shape[1]

    center = (w // 2, h // 2)
    r = cv2.getRotationMatrix2D(center, angle, 1)
    # 調整旋轉後的長寬
    rotated_h = int((w * np.abs(r[0, 1]) + (h * np.abs(r[0, 0]))))
    rotated_w = int((h * np.abs(r[0, 1]) + (w * np.abs(r[0, 0]))))
    r[0, 2] += (rotated_w - w) // 2
    r[1, 2] += (rotated_h - h) // 2
    # 旋轉圖像
    rotated_img = cv2.warpAffine(src, r, (rotated_w, rotated_h))
    return rotated_img

all_begin_time = time.time()
all_start_time = all_begin_time

MIN_MATCH_COUNT = 10 # 設置最低特征點匹配數量為10
template = cv2.imread('red-base1.jpg')  # queryImage
#target = cv2.imread('./folded20_4/unfloded_source.jpg')   # trainImage
target = cv2.imread('F:/project/bottlecap/red/ok/Image_20240202162541518.jpg') #('./cap/1red-5000.jpg')   # trainImage
#target = cv2.imread('F:/project/Golf/Data/34.jpg')   # trainImage

h, w, _ = target.shape
#target = target[0:h, 0:int((w/2)+(w/4))].copy()

target1 = target.copy()
template1 = template.copy()

# gray= cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

# Initiate SIFT detector創建sift檢測器
sift = cv2.SIFT_create()    # cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(template, None)
kp2, des2 = sift.detectAndCompute(target, None)

# 標示關鍵點
# flags：绘图功能的标识设置
# cv2.DRAW_MATCHES_FLAGS_DEFAULT：创建输出图像矩阵，使用现存的输出图像绘制匹配对和特征点，对每一个关键点只绘制中间点
# cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG：不创建输出图像矩阵，而是在输出图像上绘制匹配对
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS：常用--对每一个特征点绘制带大小和方向的关键点图形
# cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS：单点的特征点不被绘制
cv2.drawKeypoints(template1, kp1, template1, (0,255,0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('keyPoint1', template1)
cv2.drawKeypoints(target1, kp2, target1, (0,255,0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('keyPoint2', target1)

# 創建設置FLANN匹配
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

# orb需降低opencv版本
# orb = cv2.ORB_create(nfeatures=500)
# kp1, des1 = orb.detectAndCompute(template, None)
# kp2, des2 = orb.detectAndCompute(target, None)
# FLANN_INDEX_LSH = 6
# index_params = dict(algorithm=FLANN_INDEX_LSH,
#                    table_number=6,
#                    key_size=12,
#                    multi_probe_level=1)
# search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# store all the good matches as per Lowe's ratio test.
matchType = type(matches[0])
#print(matchType)
good = []
if isinstance(matches[0], cv2.DMatch):
    good = matches
else:
    # 捨棄大於0.7的匹配
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
'''
现在我们設置只有存在 10 个以上匹配时才去查找目标 MIN_MATCH_COUNT=10   否则显示報告消息  现在匹配不足   
如果找到了足够的匹配  會提取两幅图像中匹配点的坐标。把它们传入到函数中計算和变换。一旦我们找到 3x3 的变换矩陣就可以使用它将查找图像的四个点变换到目标图像中去了。然后再绘制出来。
'''
# print(len(good))
if len(good)>=MIN_MATCH_COUNT:
    # 獲取關鍵點的坐標
    # type: cv2.DMatch
    # DMatch屬性
    #   distance: 描述符之間的距離，越小越好
    #   trainIdx: 目標圖像中的描述符索引
    #   queryIdx: 查詢圖像中的描述符索引
    #   imgIdx: 目標圖像的索引
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    # print([kp1[m.queryIdx].angle for m in good])
    # print([kp2[m.trainIdx].angle for m in good])
    src_size = [kp1[m.queryIdx].size for m in good]
    dst_size = [kp2[m.trainIdx].size for m in good]
    src_angle = [kp1[m.queryIdx].angle for m in good]
    dst_angle = [kp2[m.trainIdx].angle for m in good]
    # print(len(src_angle), src_angle)
    # print(len(dst_angle), dst_angle)
    # print(len(src_size), src_size)
    # print(len(dst_size), dst_size)
    #print(m.distance, m.queryIdx, m.trainIdx, m.imgIdx)
    # 第三个参数 Method used to computed a homography matrix. The following methods are possible: #0 - a regular method using all the points
    # CV_RANSAC - RANSAC-based robust method
    # CV_LMEDS - Least-Median robust method
    # 第四个参数取值范围在 1 到 10  绝一个点对的 值。
    # 傳回值中 M 为变换矩陣。
    # 計算變換矩陣和MASK
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) # cv2.RANSAC cv2.LMEDS
    matchesMask = mask.ravel().tolist()
    # print(M)
    # print(mask)
    print(len(matchesMask), matchesMask)
    h1 = template.shape[0]
    w1 = template.shape[1]
    # 使用得到的變換矩陣對原圖像的四個角進行變換，獲得在目標圖像上對應的坐標
    pts = np.float32([[0, 0], [0, h1-1], [w1-1, h1-1], [w1-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    # print(dst)
    r_img = target.copy()
    target = cv2.polylines(target, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)
else:
    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
    matchesMask = None
    print('Please press ank key to stop!')
    
draw_params = dict(matchColor=(0, 255, 0),      # draw matches in green color
                   singlePointColor=None,
                   matchesMask=matchesMask,     # draw only inliers
                   flags=2)
result = cv2.drawMatches(template, kp1, target, kp2, good, None, **draw_params)
# 關鍵點位置 kp1.pt
# 關鍵點尺寸 kp1.size
# 關鍵點角度 kp1.angle
# print(len(good), kp1[good[10].queryIdx].angle, kp2[good[10].trainIdx].angle)
# check and rotated pic
if matchesMask != None:
    cmp_size = 999
    for i in range(len(matchesMask)):
        if src_size[i] < cmp_size and matchesMask[i] > 0:
            angle = dst_angle[i] - src_angle[i]
            cmp_size = src_size[i]
            print(i, end=' ')
            # break
    # angle = kp2[good[4].trainIdx].angle - kp1[good[4].queryIdx].angle
    r_img = img_rotate(r_img, angle)
    cv2.imwrite('./temp/rotated.jpg', r_img)
    r_img = img_rotate(target, angle)
    cv2.putText(r_img, "Rotated angle: %.3f" % angle, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
    cv2.imshow('R', r_img)

    #plt.imshow(result, 'gray')
    #plt.show()
    cv2.imshow('match', result)
    cv2.imwrite('./temp/FLANN.jpg', result)
    cv2.imwrite('./temp/rotated_inf.jpg', r_img)
    all_end_time = time.time()
    print("\n執行時間：%.3f 秒" % (all_end_time - all_begin_time))

cv2.waitKey(0)
cv2.destroyAllWindows()
