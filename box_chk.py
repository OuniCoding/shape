#!/usr/bin/python
'''
自動檢測邏輯
1。偵測瓶蓋
    使用 cv2.HoughCircles（也可以切換 SimpleBlobDetector）檢測所有圓形瓶蓋。
    半徑範圍用 CAP_RADIUS_RANGE 設定。
2。自動計算格距與方向
    對所有檢測到的圓心做 DBSCAN 分群找相鄰瓶蓋距離。
    找出最常出現的水平距離與垂直距離（對應蜂巢間距）。
    同時計算旋轉角度，確保格點可以傾斜擺放。
3。自動定位第一格中心
    找出最左上角的瓶蓋位置，當作左上第一格。
4。建立 6x6 蜂巢格位座標
    根據 pitch 與蜂巢偏移規則生成理想格位。
5。比對格位與實際檢測結果
    判斷缺瓶與多餘瓶，並畫出結果。

使用方法
把 IMAGE_PATH 換成你的照片路徑
設定 CAP_RADIUS_RANGE 為瓶蓋的像素半徑範圍
例如瓶蓋直徑 40px → 半徑 20 → 設 (18, 22)
直接執行，不需要點座標，也不用輸入格距
輸出缺瓶數與多餘瓶數，並在影像上用：
紅叉 → 缺瓶格位
黃圈 → 多餘瓶
綠圈 → 正常瓶
'''

import cv2
import numpy as np
import math
from collections import Counter
from sklearn.cluster import DBSCAN

# ===== 使用者設定 =====
IMAGE_PATH = "box.jpg"         # 圖片路徑
ROWS, COLS = 6, 6               # 蜂巢排列
CAP_RADIUS_RANGE = (18, 45)     # 瓶蓋半徑範圍（像素）
OCCUPY_DIST_RATIO = 0.55        # 占用距離比例

# ---- 偵測瓶蓋 ----
def detect_caps(gray, rmin, rmax, p1=100, p2=25):
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=rmin*0.8,
        param1=p1, param2=p2, minRadius=rmin, maxRadius=rmax
    )
    if circles is None:
        return []
    circles = np.round(circles[0, :]).astype(int)
    return [(x, y, r) for x, y, r in circles]

# ---- 計算格距與角度 ----
def estimate_pitch_and_angle(centers):
    centers = np.array(centers)
    dists = []
    angles = []
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            dx = centers[j][0] - centers[i][0]
            dy = centers[j][1] - centers[i][1]
            dist = math.hypot(dx, dy)
            if 10 < dist < 500:  # 排除太近與太遠
                dists.append(dist)
                angles.append(math.atan2(dy, dx))
    # 水平距離取眾數
    dist_counts = Counter(np.round(dists, 1))
    pitch_x = dist_counts.most_common(1)[0][0]
    # 角度取中位數
    angle = np.median(angles)
    # 垂直距離 = pitch_x * sqrt(3)/2 (蜂巢規則)
    pitch_y = pitch_x * math.sqrt(3) / 2
    return pitch_x, pitch_y, angle

# ---- 建立蜂巢格位 ----
def build_honeycomb_centers(origin, pitch_x, pitch_y, rows, cols, angle):
    cos_t, sin_t = math.cos(angle), math.sin(angle)
    ux, uy = cos_t * pitch_x, sin_t * pitch_x
    vx, vy = -sin_t * pitch_y, cos_t * pitch_y
    centers = []
    for r in range(rows):
        offset = 0.5 if (r % 2 == 1) else 0.0
        base = np.array(origin, dtype=float) + r * np.array([vx, vy]) + offset * np.array([ux, uy])
        for c in range(cols):
            p = base + c * np.array([ux, uy])
            centers.append(tuple(p))
    return centers

# ---- 配對 ----
def nearest_match(expected_pts, detected_pts, max_dist):
    matches = {}
    used_det = set()
    for i, (ex, ey) in enumerate(expected_pts):
        min_d, min_j = float('inf'), None
        for j, (dx, dy, _) in enumerate(detected_pts):
            if j in used_det:
                continue
            dist = math.hypot(dx - ex, dy - ey)
            if dist < min_d:
                min_d, min_j = dist, j
        if min_d <= max_dist:
            matches[i] = min_j
            used_det.add(min_j)
    unmatched_expected = set(range(len(expected_pts))) - set(matches.keys())
    unmatched_detected = set(range(len(detected_pts))) - used_det
    return matches, unmatched_expected, unmatched_detected

# ---- 主程式 ----
def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"找不到影像：{IMAGE_PATH}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # 偵測瓶蓋
    caps = detect_caps(gray, CAP_RADIUS_RANGE[0], CAP_RADIUS_RANGE[1])
    centers = [(x, y) for x, y, _ in caps]
    if not centers:
        print("沒有檢測到瓶蓋！")
        return

    # 推算格距與角度
    pitch_x, pitch_y, angle = estimate_pitch_and_angle(centers)

    # 找左上瓶蓋作為第一格
    first_center = min(centers, key=lambda p: (p[1], p[0]))

    # 生成格位
    expected_pts = build_honeycomb_centers(first_center, pitch_x, pitch_y, ROWS, COLS, angle)

    # 匹配
    max_dist = pitch_x * OCCUPY_DIST_RATIO
    matches, missing_idx, extra_idx = nearest_match(expected_pts, caps, max_dist)

    # 視覺化
    vis = img.copy()
    for (x, y) in expected_pts:
        cv2.circle(vis, (int(x), int(y)), 4, (200, 200, 200), -1)
    for i in missing_idx:
        x, y = expected_pts[i]
        cv2.line(vis, (int(x)-10, int(y)-10), (int(x)+10, int(y)+10), (0, 0, 255), 3)
        cv2.line(vis, (int(x)-10, int(y)+10), (int(x)+10, int(y)-10), (0, 0, 255), 3)
    for j, (x, y, r) in enumerate(caps):
        color = (0, 255, 0) if j not in extra_idx else (0, 255, 255)
        cv2.circle(vis, (x, y), r, color, 2)

    print("缺瓶數：", len(missing_idx))
    print("多餘瓶數：", len(extra_idx))
    cv2.imshow("result", vis)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
