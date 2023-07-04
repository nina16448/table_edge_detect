# 在使用者打擊時，才需要傳遞位置
# 犯規時也需要傳遞初始位置

def cueball_state_judge(detections):
    max_conf = -1
    max_conf_pos = None

    for det in detections:
        if len(det):
            for *xyxy, conf, cls in det:
                if int(cls) == 1 and conf > max_conf:
                    max_conf = conf
                    x1, y1, x2, y2 = xyxy
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    max_conf_pos = (center_x.item(), center_y.item())

    return max_conf_pos