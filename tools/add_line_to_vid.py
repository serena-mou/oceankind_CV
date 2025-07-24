import os
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = str(2**18)

import cv2
import numpy as np
save = True
src = "/home/serena/Data/Urchins/manual_count/GH016323.MP4"
cap = cv2.VideoCapture(src)
name = "line"
pix_from_bottom = 300
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