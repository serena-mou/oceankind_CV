#!/usr/bin/env python3

"""
Author: Serena Mou
Created:  17 October 2024

===
Visualise YOLO segmentation annotations
===

"""
import argparse
import cv2
import os
import yaml
import glob
import sys
import numpy as np

def arg_parse():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Visualise segmentation annotations')

    parser.add_argument("--src", dest = "src",
            help = "Path to dataset root folder", default = None, type = str, required=True)
    parser.add_argument("--img_folder", dest = "img_folder",
            help = "Name of folder to images in --src", default = "all_images", type = str)
    parser.add_argument("--label_folder", dest = "label_folder",
            help = "Name of folder to labels in --src", default = "all_labels", type = str)
    parser.add_argument("--data", dest = "data",
            help = "Name of yaml file with list of classes", default = "data.yaml", type = str)
    parser.add_argument("--scale", dest = "scale", 
            help = "Fraction to scale images to. Default 0.5", default = 0.5, type = float)


    return parser.parse_args()

def seg_converter(seg, im_sz):
## Convert box in yolo format to cv2 rectangle format (top left x, top left y), (bottom right x, bottom right y) in pixels
    seg = seg.split(' ')
    # print(im_sz)
    scale = np.array([im_sz[1], im_sz[0]])
    # print(scale)
    data = [float(s) for s in seg]
    l = int(len(data)/2)
    data = np.reshape(data,(l,2))*scale
    # print(data.astype(int))
    return data.astype(int)

def colors(cls):
    # 22 distinct colors. if class number is more than 22, wrap around
    colors = [(75, 25, 230), (75, 180, 60), (25, 225, 255), (200, 130, 0), (48, 130, 245), 
              (180, 30, 145), (240, 240, 70), (230, 50, 240), (60, 245, 210), (212, 190, 250), 
              (128, 128, 0), (255, 190, 220), (40, 110, 170), (200, 250, 255), (0, 0, 128), 
              (195, 255, 170), (0, 128, 128), (180, 215, 255), (128, 0, 0), (128, 128, 128), 
              (255, 255, 255), (0, 0, 0)]
    return colors[(cls+len(colors))%len(colors)]


def main():
    args = arg_parse()

    # load classes into dict
    with open(os.path.join(args.src,args.data), 'r') as f:
        data_loaded = yaml.safe_load(f)
    classes = data_loaded["names"]

    # load labels
    labels = sorted(glob.glob(os.path.join(args.src, args.label_folder,'*')))
    print("Press any key to go to the next image, press ESC to escape.")
    # for each label
    # for label in labels:
    for i in range(0,len(labels)):
    # for i in range(len(labels)):
        label = labels[i]
        # get matching image

        # img name
        label_name = label.split('/')[-1]
        img_name = label_name.split('.')[0]
        img_path = glob.glob(os.path.join(args.src, args.img_folder,img_name+"*"))
        print(label)
        if len(img_path) > 1:
            print("WARNING: label name matched to more than one img")
        im = cv2.resize(cv2.imread(img_path[0]), (0,0), fx=args.scale, fy=args.scale)

        # for each label in the label file, draw a box and the class
        f = open(label,"r")
        for line in f:
            line.rstrip()
            first_space = line.find(" ")
            cls = int(line[0:first_space])
            box = line[first_space+1:]
            # print(label_name)
            seg = seg_converter(box, im.shape)
            for s in seg:
                cv2.circle(im, (s[0],s[1]),5,color=colors(3),thickness=-1)
                # im[s[0],s[1],:] = colors(cls)

        try:
            cv2.imshow(label_name,im)
            key = cv2.waitKey(0)
            if key == 27:
                sys.exit()    
            cv2.destroyAllWindows()
        except KeyboardInterrupt:
            sys.exit()
if __name__=='__main__':
    main()
