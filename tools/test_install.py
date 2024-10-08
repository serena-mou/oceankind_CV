from ultralytics import YOLO
import torch

## Check that ultralytics is installed and CUDA is available

model = YOLO("yolov8n.pt")
print("Is CUDA available?", torch.cuda.is_available())