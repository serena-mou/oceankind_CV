# Run patch model on full images, visualising output
# Written by: Serena Mou
# Date:       4th Aug 2026

import os
import getch
import cv2
import glob
import shutil
import numpy as np
import argparse
import sys
from tqdm import tqdm
from ultralytics import YOLO
from csv import writer
import pandas as pd


def arg_parse():

    parser = argparse.ArgumentParser(
        description="Run patch classification model"
    )

    parser.add_argument("-src","--source", dest="source",
        help = "Path to folder of images to run.", default=None, type=str, required=True
    )
    parser.add_argument("-s","--save", dest="save",
        help = "Path folder to save.", default=None, type=str, required=True
    )
    parser.add_argument("-sh","--shape", dest="shape",
        help = "Shape of output in x,y. Default 5x5", default="5,5", type=str, required=False
    )
    parser.add_argument("-w","--weights", dest="weights",
        help = "Path to model weights.", default=None, type=str, required=True
    )

    
    return parser.parse_args()



class visPatches():
    def __init__(self, src, save, shape, weights):
        # init things
        # load and sort all images in path
        # TODO other filetypes?

        self.all_ims = sorted(glob.glob(os.path.join(src,"*.jpg")))
        if len(self.all_ims) == 0:
            sys.exit("No .jpg files found in source. Check path.")

        # make output folder
        os.makedirs(save, exist_ok=True)

        # parse save location 
        self.save = save
       
        # load patch shape
        self.shape = [int(x) for x in shape.split(',')]
        
        # load model
        self.model = YOLO(weights, task='classify')
        self.class_dict = self.model.names
        # list of colours in BGR [blue, red, yellow, green, purple]
        self.colours = [(200,130,0),(75,25,230),(25,255,255),(75,180,60),(180,30,145)]
        if len(self.class_dict) > len(self.colours):
            sys.exit("Not enough colours for number of classes. Add more. Exiting.")

    def run(self):

        # load images
        for im in self.all_ims:
            image = cv2.imread(im)
            image_name = os.path.basename(im)
            #img_ref = self.img_ref.index(image_name)#self.img_dict.get(str(image_name))
            # image_name = os.path.splitext(image_name)[0]
            # image size 
            u = int(image.shape[1]/self.shape[1])
            v = int(image.shape[0]/self.shape[0])
            mask = np.zeros(image.shape) 
            # list of all crops in image
            im_list = []
            for x in range(self.shape[1]):
                for y in range(self.shape[0]):
                    x_start, y_start, x_end, y_end = int(x*u), int(y*v), int((x+1)*u), int((y+1)*v)
                    cropped_im = image[y_start:y_end, x_start:x_end]
                    im_list.append(cropped_im)
                    # cv2.imwrite(os.path.join(self.save, image_name+"_%i_%i.jpg"%(x,y)), cropped_im)
            
            # run all patches through model as batch
            results = self.model(im_list, imgsz=224, verbose=False)

            patch_summary_top = []
            for result in results: 
                patch_summary_top.append(result.probs.top1)
            
            idx = 0
            for x in range(self.shape[1]):
                for y in range(self.shape[0]):
                    patch_cls = patch_summary_top[idx]
                    x_start, y_start, x_end, y_end = int(x*u), int(y*v), int((x+1)*u), int((y+1)*v)
                    mask[y_start:y_end, x_start:x_end,:]=self.colours[patch_cls]
                    idx+=1
            mask = mask/255.0
            combined = cv2.addWeighted(image/255.0,0.85,mask,0.15,0)
            combined = np.round(combined*255).astype(np.uint8)
            # print(combined)
            # print(combined.dtype)
            # input()
            cv2.imwrite(os.path.join(self.save,"mask_"+image_name),combined)
            # cv2.imshow("weighted",cv2.resize(combined,(0,0),fx=0.2,fy=0.2))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            

def main():
    args = arg_parse()
    vP = visPatches(args.source, args.save, args.shape, args.weights)
    vP.run()

if __name__=='__main__':
    main()
