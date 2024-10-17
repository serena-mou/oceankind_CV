#!/usr/bin/env python3

"""
Author: Matthew Tavatgis, Updated by SM
Created: 9th Feb 2024

Use Ultralytics to test a YOLO model
"""

import sys
import argparse
from ultralytics import YOLO

def arg_parse():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Test YOLO Ultralytics Model')

    parser.add_argument("--src", dest = "src",
            help = "Test yaml describing test set", default = None, type = str)
    parser.add_argument("--weights", dest = "weights",
            help = "Model weights to evaluate", default = None, type = str),
    parser.add_argument("--name", dest = "name",
            help = "Name for training outputs", default = None, type = str)

    return parser.parse_args()

def main():
    # Get args
    args = arg_parse()

    # Check yaml and weights
    if args.src is None:
        print("ERROR: Test yaml must be provided with --src! Exiting...\n")
        sys.exit()

    if args.weights is None:
        print("ERROR: Model weights must be provided with --weights! Exiting...\n")
        sys.exit()

    # Select pretrained model
    model = YOLO(args.weights)

    # Test model
    metrics = model.val(data = args.src,
            project = 'OK_CV',
            name = args.name
    )

if __name__ == '__main__':
    main()
