#!/usr/bin/env python3

"""
Author: Serena Mou
Created: 11 Oct 2024

Convert data exported from label-studio into format for splitting
"""

import shutil
import argparse
import sys
import os
import yaml

def arg_parse():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Convert from label-studio into OK_CV format')

    parser.add_argument("--src", dest = "src",
            help = "Path to unzipped label-studio folder", default = None, type = str)
    parser.add_argument("--dest", dest = "dest",
            help = "Path to save resulting files", default = None, type = str)

    return parser.parse_args()


def write_yaml(classes, dest):
    # generate yaml file with each class name
    yaml_path = os.path.join(dest,"data.yaml")
    test_yaml_path = os.path.join(dest,"test.yaml")
    # dictionary of {0: class0, 1: class1...}
    cls_dict = {k:v for k,v in enumerate(classes)}

    data = {
        "path":dest,
        "train": "train",
        "val": "valid",
        "names": cls_dict
    }

    test = {
        "path":dest,
        "train": "train",
        "val": "test",
        "names": cls_dict
    }


    with open(yaml_path, 'w') as outfile:
        yaml.dump(data, outfile, sort_keys=False)

    with open(test_yaml_path, 'w') as outfile:
        yaml.dump(test, outfile, sort_keys=False)


def main():
    # Get args
    args = arg_parse()


    # Check yaml and weights
    if args.src is None:
        print("ERROR: Source folder must be provided with --src! Exiting...\n")
        sys.exit()

    if args.dest is None:
        print("ERROR: Destination folder must be provided with --dest! Exiting...\n")
        sys.exit()

    # create destination folder if not already existing
    os.makedirs(args.dest, exist_ok=True)

    # copy images and rename folder
    shutil.copytree(os.path.join(args.src, "images"), os.path.join(args.dest,"all_images"))
    # copy labels and rename folder
    shutil.copytree(os.path.join(args.src, "labels"), os.path.join(args.dest,"all_labels"))

    # generate data.yaml and test.yaml
    cls = os.path.join(args.src, "classes.txt")
    cls_list = []
    with open(cls, "r") as f:
        for x in f:
            cls_list.append(x.strip())

    write_yaml(cls_list, args.dest)
    
if __name__ == '__main__':
    main()

