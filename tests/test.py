import sys
from pathlib import Path



# Add the root directory to Python's path
sys.path.insert(0, str(Path(__file__).parent.parent))
print(str(Path(__file__).parent.parent))
import betterercam
print(betterercam.__file__)  # Should point to your local `betterercam` directory

# camera = betterercam.create(nvidia_gpu=True)

# frame = camera.grab()
# print(frame.shape)
# print(type(frame))

import os
import time
import cv2
# import dxcam
import cupy as cp



screen_x = 2560
screen_y = 1440
h_w_capture = (320,320)
x_offset = (screen_x - h_w_capture[1])//2
y_offset = (screen_y - h_w_capture[0])//2
capture_region = (0 + x_offset, 0 + y_offset, screen_x - x_offset, screen_y - y_offset)
window_height, window_width = h_w_capture

print(capture_region[3] - capture_region[1], capture_region[2] - capture_region[0])
# camera = dxcam.create(region = capture_region)
camera = betterercam.create(region=  capture_region, output_color="BGR", nvidia_gpu=True)

# cv2.namedWindow("Screen Capture", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Screen Capture", window_width, window_height)

last_fps_update = time.perf_counter()
frame_count = 0
test_frame = camera.grab()
print(test_frame.shape)
print(type(test_frame))
while True:
    
    frame = camera.grab()
    if frame is None:
        continue
    
    # cv2.imshow("Screen Capture", cp.asnumpy(frame))
    # cv2.waitKey(1)
    frame_count+=1
    current_time = time.perf_counter()
    if current_time - last_fps_update >= 1:
        fps = frame_count / (current_time - last_fps_update)
        last_fps_update = current_time
        frame_count = 0
        print(f'fps: {fps:.2f}')
