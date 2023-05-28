import sys
import cv2
import numpy as np
import edge_detect as ED


def sort_points(points):
    points = np.array(points)  # 将 points 转换为 NumPy 数组
    sorted_points = np.zeros((4, 2), dtype=np.float32)
    sum_points = points.sum(axis=1)
    sorted_points[0] = points[np.argmin(sum_points)]
    sorted_points[2] = points[np.argmax(sum_points)]
    diff_points = np.diff(points, axis=1)
    sorted_points[1] = points[np.argmin(diff_points)]
    sorted_points[3] = points[np.argmax(diff_points)]
    return sorted_points


def mouse_click(event, x, y, flags, param):
    global img, img_original, intersections
    if event == cv2.EVENT_LBUTTONDOWN:
        # 計算與用戶點擊的點距離最近的交點
        min_distance = float("inf")
        min_index = -1
        for i, intersection in enumerate(intersections):
            distance = np.sqrt((x - intersection[0]) ** 2 + (y - intersection[1]) ** 2)
            if distance < min_distance:
                min_distance = distance
                min_index = i

        # 用戶點擊的點替換原始交點
        intersections[min_index] = np.array([x, y])
        print("new point at()", x, ",", y, ")")
        # 重繪邊界和交點
        img = img_original.copy()
        ED.draw_lines_between_points(img, intersections)


def prev_exe(pic):
    global img, img_original, intersections
    img, img_original, intersections, flag = ED.get_edge(pic)
    click_count = 0

    if flag == False:
        return img, intersections, False

    cv2.namedWindow("Intersection points")
    cv2.setMouseCallback("Intersection points", mouse_click)
    while True:
        temp_img = img.copy()
        for point in intersections:
            cv2.circle(temp_img, tuple(point.astype(int)), 5, (0, 0, 255), 4)
        cv2.putText(
            temp_img,
            "Press the 's' to confirm, press the 'r' to restore, and press the 'q' to exit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.imshow("Intersection points", temp_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            confirmed = True
            break
        elif key == ord("q"):
            confirmed = False
            break
        elif key == ord("r"):
            img, img_original, intersections = ED.get_edge(pic)

    if confirmed:
        cv2.destroyAllWindows()
        # 將圖片透視變形成正上方的視角
        intersections = sort_points(intersections)

        width, height = 1000, 600  # 設置目標圖像的大小
        dst_points = np.array(
            [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
        )
        img = img_original.copy()
        # 計算透視變換矩陣
        M = cv2.getPerspectiveTransform(intersections, dst_points)

        # 進行透視變換
        warped_img = cv2.warpPerspective(img, M, (width, height))

        return warped_img, intersections, True

    else:
        print("User did not confirm the selection.")
        # 您可以選擇退出程序或執行其他操作
        sys.exit()  # 如果需要退出程序，添加此行


def exe_pic(P, intersections):
    # 縮小圖像
    imge = cv2.resize(P, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
    # 檢查圖像的形狀
    height, width, channels = imge.shape
    # 如果長度大於寬度，則旋轉圖像
    if height > width:
        imge = cv2.rotate(imge, cv2.ROTATE_90_CLOCKWISE)

    intersections = sort_points(intersections)

    width, height = 1000, 600  # 設置目標圖像的大小
    dst_points = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    )

    # 計算透視變換矩陣
    M = cv2.getPerspectiveTransform(intersections, dst_points)

    # 進行透視變換
    warped_img = cv2.warpPerspective(imge, M, (width, height))

    return warped_img
