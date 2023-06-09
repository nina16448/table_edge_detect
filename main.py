import edge_detect as ED
import Transformation as TS
import sys
import cv2
import numpy as np
import torch
import sand_info as SI


def main():
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    model = torch.hub.load("ultralytics/yolov5", "custom", path="../v8_m.pt")
    # 設定 IoU 門檻值
    model.iou = 0.1

    # 設定信心門檻值
    model.conf = 0.1

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

        height, width, channels = frame.shape

        if not ret: 
            break  # 如果讀不到幀，退出循環

        # 對每一幀執行邊緣提取和透視變換
        if flag == False:
            warped_frame, intersections, flag = TS.prev_exe(frame)
        else:
            warped_frame = TS.exe_pic(frame, intersections)
            warped_frame = cv2.filter2D(warped_frame, -1, kernel)

        warped_frame = cv2.resize(
            warped_frame, (640, 640), interpolation=cv2.INTER_AREA
        )

        results = model(warped_frame)
        # 將檢測結果繪製到影像上
        if len(results.xyxy) > 0:
            warped_frame = results.render()[0]

        warped_frame = cv2.resize(
            warped_frame, (1280, 640), interpolation=cv2.INTER_AREA
        )

        # 顯示變換後的幀
        cv2.imshow("Warped Frame", warped_frame)
        results.print()  # results 是 yolo v5 回傳的結果，有

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
