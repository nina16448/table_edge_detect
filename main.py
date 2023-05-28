import edge_detect as ED
import Transformation as TS
import sys
import cv2
import numpy as np


def main():
    input_video_path = "../test5.mp4"
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open the video file.")
        sys.exit()
    flag = False
    intersections = np.zeros((4, 2), dtype=np.float32)
    counter = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break  # 如果讀不到幀，退出循環

        # 對每一幀執行邊緣提取和透視變換
        if flag == False:
            warped_frame, intersections, flag = TS.prev_exe(frame)
        else:
            warped_frame = TS.exe_pic(frame, intersections)

        # 顯示變換後的幀
        cv2.imshow("Warped Frame", warped_frame)

        # 等待 1 毫秒，並檢查是否有按下 q 鍵，如果有，則退出循環
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        print(counter)
        counter += 1

    # 釋放資源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
