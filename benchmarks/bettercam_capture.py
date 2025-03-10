import time
import betterercam


TOP = 0
LEFT = 0
RIGHT = 1920
BOTTOM = 1080
region = (LEFT, TOP, RIGHT, BOTTOM)
title = "[BettererCam] Capture benchmark"

fps = 0
camera = betterercam.create(output_idx=0, output_color="BGRA")
camera.start(target_fps=60, video_mode=True)
for i in range(1000):
    image = camera.get_latest_frame()
camera.stop()
del camera
