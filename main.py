import torch
from utils import detect
import cv2
import time
start_time = time.time()

if __name__ == '__main__':
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
        text_color = (255,255,0)
        implement_half_precision = True
        save_dir = "test_output/"
        source = '0'  # [path to image/video] OR ['0' for camera on PC] 
        weights = "model_weights/End-to-end.pth"

    opt = vars_
    with torch.no_grad():
        if opt.source != '0':
            detect(opt)
        else:
            for img, angle in detect(opt):
                print(angle)
                cv2.imshow('image', img)
                cv2.waitKey(1)  # 1 millisecond
        
print("--- %s seconds ---" % (time.time() - start_time))

