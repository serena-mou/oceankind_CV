#!/usr/bin/env python3

"""
Author: Serena Mou
Created:  17 October 2024

===
Visualise and save labels from YOLO model
===

"""
import argparse
import cv2
import os
import yaml
import glob
import sys
from ultralytics import YOLO

def arg_parse():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Visualise bounding box annotations')

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
    parser.add_argument("--save", dest = "save", 
            help = "Save ims or just view. Default False", default = False, type = bool)
    parser.add_argument("--model", dest = "model", 
            help = "Model path for predictions", type = str)


    return parser.parse_args()

def bbx_converter(box, im_sz):
## Convert box in yolo format to cv2 rectangle format (top left x, top left y), (bottom right x, bottom right y) in pixels
    data = box.split(" ")
    xc = float(data[0])
    yc = float(data[1])
    bw = float(data[2])
    bh = float(data[3])
    h = int(im_sz[0])
    w = int(im_sz[1])

    tlx = int((xc*w)-(0.5*bw*w))
    tly = int((yc*h)-(0.5*bh*h))

    brx = int((xc*w)+(0.5*bw*w))
    bry = int((yc*h)+(0.5*bh*h))

    return (tlx,tly),(brx,bry)

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
    labels = glob.glob(os.path.join(args.src, args.label_folder,'*'))
    print("Press any key to go to the next image, press ESC to escape.")
    
    # create save folder
    if args.save:
        save_path_pred = os.path.join(args.src,"vis_output","model")
        os.makedirs(save_path_pred, exist_ok=True)

    # load model
    model = YOLO(args.model)
    
    # for each label
    # for label in labels:
    # for i in range(0,len(labels),10):
    for i in range(len(labels)):
        label = labels[i]
        # get matching image

        # img name
        label_name = label.split('/')[-1]
        img_name = label_name.split('.')[0]
        img_path = glob.glob(os.path.join(args.src, args.img_folder,img_name+"*"))
        if len(img_path) > 1:
            print("WARNING: label name matched to more than one img")
        im = cv2.imread(img_path[0])
        im_scale = cv2.resize(im, (0,0), fx=args.scale, fy=args.scale)

        # run model
        results = model.predict(im)
        pred_im = results[0].plot()
        #pred_rgb = cv2.cvtColor(pred_im, cv2.COLOR_BGR2RGB)
        # for each label in the label file, draw a box and the class
        try:
            if args.save:
                cv2.imwrite(os.path.join(save_path_pred,img_name+'.png'),pred_im)
                #cv2.imwrite(os.path.join(save_path_label,img_name+".png"),im)
                
            # cv2.imshow(label_name,im)
            # key = cv2.waitKey(0)
            # if key == 27:
            #     sys.exit()    
            # cv2.destroyAllWindows()
        except KeyboardInterrupt:
            sys.exit()
if __name__=='__main__':
    main()
