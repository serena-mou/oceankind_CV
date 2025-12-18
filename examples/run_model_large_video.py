# Run model on a source, can be a folder of images or video file


from ultralytics import YOLO 
import argparse
import os


parser = argparse.ArgumentParser(description='Run a pretrained model on a source')
 
parser.add_argument("--src", dest = "src",
        help = "Source directory to parse", default = None, type = str)
parser.add_argument("--model", dest = "model",
        help = "Path to model", default = None, type = str)
parser.add_argument("--name", dest = "name",
        help = "Name output folder", default = None, type = str)
parser.add_argument("--stride", dest = "stride",
        help = "Number of video frames to skip. Default 3", default = 3, type = int)
parser.add_argument("--conf", dest = "conf",
        help = "Confidence threshold. Between 0-1. Default 0.25", default = 0.25, type = float)
parser.add_argument("--device", dest = "device",
        help = "Device to run. Default 0 for CUDA enabled. Change to mps for MacOS", default = 0, type = str)

args = parser.parse_args()


if args.device == "0":
    args.device = 0

model = YOLO(args.model)    # pretrained YOLOv8n model 
results = model.predict(source=args.src,
                        vid_stride=args.stride,
                        classes = None,
                        conf=args.conf,
                        device=args.device, 
                        stream=True)

os.makedirs(args.name, exist_ok=True)

vid_name = args.src.split('/')[-1]
vid_idx = vid_name.rfind('.')
vid_name = vid_name[:vid_idx]

for i,result in enumerate(results):
        frame_no = i*args.stride
        frame_name = "%s-frame-%i.jpg"%(vid_name,frame_no) 
        save_path = os.path.join(args.name, frame_name)
        result.save(filename=save_path)
         
