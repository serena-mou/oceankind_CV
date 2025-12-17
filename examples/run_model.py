# Run model on a source, can be a folder of images or video file


from ultralytics import YOLO 
import argparse



parser = argparse.ArgumentParser(description='Run a pretrained model on a source')
 
parser.add_argument("--src", dest = "src",
        help = "Source directory to parse", default = None, type = str)
parser.add_argument("--model", dest = "model",
        help = "Path to model", default = None, type = str)
parser.add_argument("--name", dest = "name",
        help = "Name output folder", default = None, type = str)

args = parser.parse_args()

model = YOLO(args.model)    # pretrained YOLOv8n model 
results = model.predict(source=args.src,
                        show=False,
                        classes = None,
                        name=args.name,
                        conf=0.5,
                        device=0, 
                        save=True,       
                        save_txt=True )     
