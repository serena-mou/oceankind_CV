# Draw a line on a video and save output video. Used for manually counting objects in a video
# Wriiten: Serena Mou 
# Date:    9/9/25

import os
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = str(2**18)

import cv2
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Draw a line on a video and save. ')
 
parser.add_argument("--video", dest = "video",
        help = "Path to video", default = None, type = str)
parser.add_argument("--pixels", dest = "pixels",
        help = "Number of pixels from the bottom to draw line. Default 300", default = 300, type = int)

args = parser.parse_args()

save = True
src = args.video
cap = cv2.VideoCapture(src)
name = "line"
pix_from_bottom = args.pixels
color = (10,10,255)
if save:
    save_name = src.split('/')
    save_name = save_name[-1]
    save_name = "%s-p%i-%s"%(name,pix_from_bottom,save_name)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(save_name, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
 
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
               
        points = (0,frame_height-pix_from_bottom),(frame_width, frame_height-pix_from_bottom)
        cv2.line(frame, points[0], points[1], color=color, thickness=5)
        if save:
            out.write(frame)

        #cv2.imshow("ahh", cv2.resize(frame, (0,0), fx=0.2,fy=0.2))
        #cv2.waitKey(0)
 
cap.release()
out.release()
cv2.destroyAllWindows()