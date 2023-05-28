import sys
import cv2
import numpy as np


def draw_lines_between_points(img, points):
    for i, point1 in enumerate(points):
        for j, point2 in enumerate(points):
            if i != j:
                cv2.line(
                    img,
                    tuple(point1.astype(int)),
                    tuple(point2.astype(int)),
                    (255, 255, 255),
                    4,
                )


def select_line(lines, img_center, state):  # 選最中心的線
    if len(lines) == 0:
        return (-7510, -7510, -7510, -7510)
    elif len(lines) <= 1:
        # 如果只有一條直線，直接返回該直線
        return lines[0]
    else:
        min_distance = float("inf")
        selected_line = None

        for line in lines:
            # 計算每條直線的中心點
            center_x = (line[0] + line[2]) / 2
            center_y = (line[1] + line[3]) / 2

            # 計算該中心點與圖片中心點之間的距離
            if state == 1:
                distance = img_center[1] - center_y

            else:
                distance = img_center[0] - center_x
            # 更新最小距離和相應的線條
            if abs(distance) < min_distance:
                min_distance = abs(distance)
                selected_line = line
            print("??:", state, abs(distance), min_distance)
        print("!!")
        return selected_line


def line_filter(lines, state, threshold):
    filtered_lines = []
    distance_threshold = 10
    angle_threshold = threshold

    if state == 0 or state == 1:  # 上方或下方線段
        y_coords = []
        for line in lines:
            x1, y1, x2, y2 = line
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            y_coords.extend([y1, y2])

        avg_y = np.mean(y_coords)

        for line in lines:
            x1, y1, x2, y2 = line
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            print("angle: ", angle)
            if state == 0 and (y1 + y2) / 2 > avg_y:
                if abs(angle - 90) < angle_threshold:
                    filtered_lines.append(line)
            elif state == 1 and (y1 + y2) / 2 < avg_y:
                if abs(angle - 90) < angle_threshold:
                    filtered_lines.append(line)

    elif state == 2 or state == 3:  # 左方或右方線段
        x_coords = []
        for line in lines:
            x1, y1, x2, y2 = line
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            x_coords.extend([x1, x2])

        avg_x = np.mean(x_coords)

        for line in lines:
            x1, y1, x2, y2 = line
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            print("angle: ", angle)
            distance = abs((x1 + x2) / 2 - avg_x)
            if distance <= distance_threshold:
                if abs(angle) < angle_threshold or abs(angle - 180) < angle_threshold:
                    filtered_lines.append(line)

    return filtered_lines


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img_bright = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img_bright


def get_edge(P):
    # 讀取圖片
    img = P
    # 縮小圖像
    # img = cv2.resize(img, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
    # 檢查圖像的形狀
    height, width, channels = img.shape
    # 如果長度大於寬度，則旋轉圖像
    if height > width:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    img_original = img.copy()

    # 調亮圖片
    img_bright = increase_brightness(img, value=50)
    # cv2.imshow("Original Image", img_bright)

    # 利用高斯濾波器和平均濾波器做平滑化處理
    blur = cv2.GaussianBlur(img_bright, (3, 3), 0)
    avg = cv2.blur(blur, (3, 3))
    # 將圖像轉換為灰度圖像
    gray = cv2.cvtColor(avg, cv2.COLOR_RGB2GRAY)

    # 二質化圖像
    # threshold_value = 150
    max_value = 255
    # ret, binary = cv2.threshold(gray, threshold_value, max_value, cv2.THRESH_BINARY)
    blockSize = 11
    C = 7
    binary = cv2.adaptiveThreshold(
        gray, max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, C
    )

    # 使用 Canny 演算法檢測邊緣
    low_threshold = 150
    high_threshold = 240
    edges = cv2.Canny(binary, low_threshold, high_threshold)
    # cv2.imshow("imshow", edges)
    # 霍夫變換檢測直線
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    print("Number of lines detected:", lines.shape[0])
    lines_up = []
    lines_down = []
    lines_left = []
    lines_right = []
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        if theta > 1 and theta < 3:
            if rho > 100:
                print("rho:", rho, " theta", theta, "  down")
                lines_down.append((a, b, x0, y0))
            else:
                print("rho:", rho, " theta", theta, "  up")
                lines_up.append((a, b, x0, y0))
        else:
            if abs(rho) > 100:
                print("rho:", rho, " theta", theta, "  right")
                lines_right.append((a, b, x0, y0))
            else:
                print("rho:", rho, " theta", theta, "  left")
                lines_left.append((a, b, x0, y0))

    print("before lines_up:", len(lines_up))
    print("before lines_down:", len(lines_down))
    print("before lines_left:", len(lines_left))
    print("before lines_right:", len(lines_right))

    angle_threshold = 0.5
    while True:
        if len(lines_up) == 0:
            break
        rec = line_filter(lines_up, 0, angle_threshold)
        angle_threshold += 0.5
        if len(rec) > 0:
            lines_up = rec
            break
    angle_threshold = 0.5
    while True:
        if len(lines_down) == 0:
            break
        rec = line_filter(lines_down, 1, angle_threshold)
        angle_threshold += 0.5
        if len(rec) > 0:
            lines_down = rec
            break
    angle_threshold = 0.5
    while True:
        if len(lines_left) == 0:
            break
        rec = line_filter(lines_left, 2, angle_threshold)
        angle_threshold += 0.5
        if len(rec) > 0:
            lines_left = rec
            break
    angle_threshold = 0.5
    while True:
        if len(lines_right) == 0:
            break
        rec = line_filter(lines_right, 3, angle_threshold)
        angle_threshold += 0.5
        if len(rec) > 0:
            lines_right = rec
            break

    print("after lines_up:", len(lines_up))
    print("after lines_down:", len(lines_down))
    print("after lines_left:", len(lines_left))
    print("after lines_right:", len(lines_right))

    img_center = (180, 180)
    # 找出邊界
    edge_line = []
    intersections = []
    edge_line.append(select_line(lines_up, img_center, 1))
    edge_line.append(select_line(lines_down, img_center, 1))
    edge_line.append(select_line(lines_left, img_center, 0))
    edge_line.append(select_line(lines_right, img_center, 0))
    edge_line = [item for item in edge_line if item != (-7510, -7510, -7510, -7510)]
    print(len(edge_line))

    # input("按下 Enter 鍵繼續...")
    flag = True
    if len(edge_line) != 4:
        flag = False
        print("Not found edge.")
        # 設定為圖片的四個角的座標點
        edge_line = [
            (0, 0, 0, height),
            (0, 0, width, 0),
            (width, height, 0, height),
            (width, height, width, 0),
        ]

        # return img, img_original, intersections, False

    center_coordinates = (int(img_center[0]), int(img_center[1]))
    print("center:", center_coordinates)

    for line in edge_line:
        a, b, x0, y0 = line
        x1 = int(x0 + 2000 * (-b))
        y1 = int(y0 + 2000 * (a))
        x2 = int(x0 - 2000 * (-b))
        y2 = int(y0 - 2000 * (a))

    # 計算四條線的方程式
    line_equations = []  #  y = mx + b ，其中 m 是斜率，b 是截距
    for line in edge_line:
        a, b, x0, y0 = line
        if b != 0:
            slope = -a / b  # 斜率
            intercept = y0 - slope * x0  # 截距 數學好難 = = 忘光ㄌ
        else:
            # 當斜率不存在時，方程式為 x = c 的形式
            slope = float("inf")
            intercept = x0
        line_equations.append((slope, intercept))

    # 求解線性方程組

    for i in range(len(line_equations)):
        for j in range(i + 1, len(line_equations)):
            eq1 = line_equations[i]
            eq2 = line_equations[j]
            if eq1[0] == eq2[0]:  # 如果兩條線的斜率相同，則跳過
                continue
            elif eq1[0] == float("inf"):
                # 斜率為無限大的垂直線的交點 x 座標設為其自身的 x 座標
                intersection_x = eq1[1]
                intersection_y = eq2[0] * intersection_x + eq2[1]
                intersection = (intersection_x, intersection_y)
            elif eq2[0] == float("inf"):
                # 斜率為無限大的垂直線的交點 x 座標設為其自身的 x 座標
                intersection_x = eq2[1]
                intersection_y = eq1[0] * intersection_x + eq1[1]
                intersection = (intersection_x, intersection_y)
            else:
                A = np.array([[eq1[0], -1], [eq2[0], -1]])
                b = np.array([-eq1[1], -eq2[1]])
                try:
                    print("find point!")
                    intersection = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    continue
            # 判斷交點是否在圖像範圍內
            if (
                intersection[0] < 0
                or intersection[0] > img.shape[1]
                or intersection[1] < 0
                or intersection[1] > img.shape[0]
            ):
                continue
            intersections.append(np.array(intersection))

    if flag == False:
        intersections = []
        # 設定為圖片的四個角的座標點
        intersections.append(np.array([0, 0]))  # 左上角
        intersections.append(np.array([img.shape[1], 0]))  # 右上角
        intersections.append(np.array([0, img.shape[0]]))  # 左下角
        intersections.append(np.array([img.shape[1], img.shape[0]]))  # 右下角

    draw_lines_between_points(img, intersections)

    # 輸出交點的位置
    print("Intersection points:")
    for point in intersections:
        print("({:.2f}, {:.2f})".format(point[0], point[1]))

    return img, img_original, intersections, True  # 返回圖像、圖像副本和交點列表
