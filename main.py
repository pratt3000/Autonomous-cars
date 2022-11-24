import torch
from utils import detect_streams, detect_static
import cv2
import time


class vars_:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        img_size = 640
        conf_thres = 0.25
        iou_thres = 0.45
        threshold = 70       # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 4  # minimum number of pixels making up a line
        max_line_gap = 70    # maximum gap in pixels between connectable line segments
        lane_boundary_top = 450 # top limit of the box in which we are considering the lane boundaries
        lane_boundary_top_offset = 450
        lane_boundary_bottom_offset = 150
        text_color = (0,0,255)
        implement_half_precision = True
        save_dir = "test_output/"
        # source = '0'  # [path to image/video] OR ['0' for camera on PC] 
        source = 'test_data/car.jpeg'
        weights = "model_weights/End-to-end.pth"

start_time = time.time()
if __name__ == '__main__':

    opt = vars_
    with torch.no_grad():
        if opt.source != '0':
            detect_static(opt)
        else:
            for img, steering_angle, obstacle_in_way in detect_streams(opt):
                print(steering_angle)
                cv2.imshow('image', img)
                cv2.waitKey(1)  # 1 millisecond
        
print("--- %s seconds ---" % (time.time() - start_time))

# Todo:
# 1. Check bboxs and return if one is too big or obj too close (use lidar inferences)
# 2. Give warning if obj close (text?)
# 3. Warning if going out of drivable region
# 4. Write car translation code
# print stop text on image