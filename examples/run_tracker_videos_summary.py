'''
Run model and tracker on a video. Counts SINGLE CLASS objects. DO NOT USE FOR MULTICLASS AT THIS POINT.
Saves a csv that summarises the detections and tracks. 
Written by: Serena Mou
Date:       18/12/25

'''

import os
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = str(2**18)

from collections import defaultdict

import cv2
import numpy as np
import time
import sys

from ultralytics import YOLO

import argparse
parser = argparse.ArgumentParser(description='Run Tracker on Video')

parser.add_argument("--src", dest = "src",
            help = "Path to video to run model on", default = None, type = str)
parser.add_argument("--model", dest = "model",
            help = "Path to model to run. Default is yolo11m", default = "yolo11m.pt", type = str)
parser.add_argument("--save", dest = "save",
            help = "Save resulting video? Default False", default = False, type = bool)
parser.add_argument("--show", dest = "show",
            help = "Show frames as they process? Default False", default = False, type = bool)
parser.add_argument("--track_len", dest = "track_len",
            help = "Number of frames detected before counting. Default 5.", default = 5, type = int)
parser.add_argument("--conf", dest = "conf",
            help = "Confidence threshold for detections. Default 0.25", default = 0.25, type = float)
parser.add_argument("--name", dest = "name",
            help = "Name to prepend to video name save. Default \"tracked\"", default = "tracked", type = str)


args = parser.parse_args()

# Load the YOLO11 model
model = YOLO(args.model)

# Open the video file
# video_path = "/home/serena/Data/Urchins/videos/GH013242.MP4"
video_path = args.src
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
if args.save:
    save_name = args.src.split('/')
    save_name = save_name[-1]
    save_name = "%s-t%i-c%.2f-%s"%(args.name,args.track_len, args.conf,save_name)
    rm_ext = save_name.rfind(".")
    csv_name = save_name[0:rm_ext]+".csv"
    if os.path.isfile(csv_name):
        ow = input("WARNING: %s file already exists. Overwrite contents? Y/N "%csv_name)
        if ow.lower() == "y":
                print("Overwriting..")
        else:
            sys.exit("ERROR: --save path already exists and overwrite NOT selected. Choose different --save path.")
    # write csv header
    header_end = ""
    for key, value in model.names.items():
        csv_str = ",%s_count,%s_track_ID"%(value,value)
        header_end+=csv_str
    
    with open(csv_name, 'w') as w:
        w.write("Frame #,Seconds,Box_x,Box_y,Box_w,Box_h,Conf%s\n"%header_end)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))  
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(save_name, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))


# Store the track history
track_history = defaultdict(lambda: [])
count = []
start = time.time()
frame_count = 0
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    frame_count +=1

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        result = model.track(frame, persist=True, show_labels=False, show_conf=False, conf=args.conf)[0]

        # Get the boxes and track IDs
        if result.boxes and result.boxes.is_track:
            boxes = result.boxes.xywh.cpu().tolist()
            track_ids = result.boxes.id.int().cpu().tolist()
            confs = result.boxes.conf.cpu().tolist()
            names = [result.names[cls.item()] for cls in result.boxes.cls.int()]
            # Visualize the result on the frame
            frame = result.plot(labels=False, probs=False)
            # Plot the tracks
            for box, track_id, conf, name in zip(boxes, track_ids, confs, names):
                x, y, w, h = box
                
                track = track_history[track_id]

                track.append((float(x), float(y)))  # x, y center point
                color = (230,230,230)
                if len(track) > args.track_len and track_id not in count:
                    count.append(track_id)
                
                if len(track) > args.track_len:
                    color = (0,255,0)
                    
                if len(track) > 30:  # retain 30 tracks for 30 frames
                    track.pop(0)
                class_id = list(model.names.keys())[list(model.names.values()).index(name)]
                pre_count = int(class_id)*2+1
                pre_commas = ","*pre_count
                with open(csv_name, 'a') as csv:
                    csv.write("%i,%.2f,%s,%s,%s,%s,%s%s%s,%s\n"%(frame_count, frame_count/fps, str(x), str(y), str(w), str(h), str(conf), pre_commas,len(count),track_id))

 
                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=color, thickness=10)

        cv2.putText(frame, "COUNT: %i"%len(count), (5,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2 )
        
        print("COUNT: ",len(count))
        # Display the annotated frame
        if args.show:
            cv2.imshow("YOLO11 Tracking", cv2.resize(frame, (0,0), fx=0.8, fy=0.8))

        if args.save:
            out.write(frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
if args.show():
    out.release()
cv2.destroyAllWindows()

time_taken = time.time()-start
print("TIME TO RUN: %.4f"%time_taken)