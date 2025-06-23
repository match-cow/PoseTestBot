import pyrealsense2 as rs
import numpy as np
import cv2

# Create a context object. This object owns the handles to all connected realsense devices
pipeline = rs.pipeline()
pipeline.start()

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        if not depth: continue

        coverage = [0]*64
        for y in range(480):
            for x in range(640):
                dist = depth.get_distance(x, y)
                if 0 < dist and dist < 1:
                    coverage[x//10] += 1
             
        for c in coverage:
            capped_c = min(c, 200)  # Cap the value of c at 200
            print(" .:nhBXWW"[capped_c // 25], end='')
    
        print()

finally:
    pipeline.stop()