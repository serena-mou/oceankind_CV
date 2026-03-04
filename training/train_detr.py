#!/usr/bin/env python3

"""
Author: Matthew Tavatgis, Updated by SM
Created: 9th Feb 2024

Use Ultralytics to train a YOLO model
"""

import sys
import argparse
from ultralytics import RTDETR

def arg_parse():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Train YOLO Model')

    parser.add_argument("--src", dest = "src",
            help = "Data yaml describing dataset", default = None, type = str)
    parser.add_argument("--name", dest = "name",
            help = "Name for training outputs", default = None, type = str)
    parser.add_argument("--classes", dest = "classes",
            help = "List of classes to train as numbers listed in data.yaml. Separate with commas ex 0,1,2", default = None, type = str)
    parser.add_argument("--pretrain", dest = "pretrain",
            help = "Pretrained model to load. Can load a different model structure or a previously trained model. Default is yolo11m.", default = "yolo11m.pt", type = str)

    return parser.parse_args()

def main():
    # Get args
    args = arg_parse()

    # Check yaml and name
    if args.src is None:
        print("ERROR: Data yaml must be provided with --src! Exiting...\n")
        sys.exit()

    if args.name is None:
        print("ERROR: Output name must be provided with --name! Exiting...\n")
        sys.exit()

    # Select pretrained model
    model = RTDETR(args.pretrain) # Pick desired default model or used pre-trained model weights. See https://docs.ultralytics.com/models/yolov8/#performance-metrics
    if args.classes is not None:
        arg_cls = args.classes.split(',')
        arg_cls = [int(x) for x in arg_cls]
    else:
        arg_cls = None
    # Start training
    metrics = model.train(data = args.src,
            epochs = 500,
            classes = arg_cls,
            patience = 50,
            batch = 1,
            imgsz = 640,
            save = True,
            device = 0,
            workers = 4,
            project = 'TURDLES',
            name = args.name,
            val = True
    )

if __name__ == '__main__':
    main()
