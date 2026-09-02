# Run patch model on luxonis full images, takes in csv with GPS location of img, adds to csv with %cover
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
        help = "Path to folder of images to run. Should include a csv of locations", default=None, type=str, required=True
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



class runPatches():
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

        csv = glob.glob(os.path.join(src, "*.csv"))
        if len(csv) > 1:
            sys.exit("More than 1 CSV found in source. Exiting. ") 
        if len(csv) == 0:
            sys.exit("No *.csv found in source. Exiting. ") 


        # copy and paste csv
        print(self.save)
        shutil.copy(csv[0], self.save)
        # read pasted csv
        csv_copied = os.path.join(self.save, os.path.basename(csv[0]))
        # read pasted copy
        self.csv_data = pd.read_csv(csv_copied, dtype=str, names=["NMEA","REF","LON","LAT","HEAD","DEPTH","INFO","HEIGHT"])
        
        # check if results output already exists, if it does, rename         
        self.csv_out = os.path.join(self.save, os.path.splitext(os.path.basename(csv[0]))[0]+'_results.csv')
        if os.path.isfile(self.csv_out):
            ext_add = 1
            root,ext = os.path.splitext(self.csv_out)
            f_new = "%s_%i%s"%(root,ext_add,ext)
            while os.path.isfile(f_new):
                ext_add +=1
                f_new = "%s_%i%s"%(root,ext_add,ext)
            self.csv_out = f_new
        
        # load Series of all images name references, cvt to list
        ref = self.csv_data.loc[:,"REF"] # image names
        self.img_ref = ref.to_list()
        
        # load patch shape
        self.shape = [int(x) for x in shape.split(',')]
        
        # load model
        self.model = YOLO(weights, task='classify')
        self.class_dict = self.model.names
        # self.chondria_top = {}
        # self.chondria_sfmx = {}
        # init dictionary where keys are [class0_top, class0_sfmx, ... classn_top, classn_sfmx]
        # values are a dictionary of {image idx: value}
        self.results_dict = {}
        for val in self.class_dict.values():
            self.results_dict[val+"_top"] = {}
            self.results_dict[val+"_sfmx"] = {}

    def run(self):

        # load images
        for im in tqdm(self.all_ims):
            image = cv2.imread(im)
            image_name = os.path.basename(im)
            img_ref = self.img_ref.index(image_name)#self.img_dict.get(str(image_name))
            # image_name = os.path.splitext(image_name)[0]
            # image size 
            u = int(image.shape[1]/self.shape[1])
            v = int(image.shape[0]/self.shape[0])
            
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

            # save results to dictionary matching class 
            for key in self.class_dict.keys():
                val = self.class_dict[key]
                patch_summary_softmax = []
                patch_summary_top = []
                for result in results: 
                    patch_summary_softmax.append(result.probs.data.detach().cpu().numpy()[key])
                    top_onehot = 1 if result.probs.top1==key else 0 
                    patch_summary_top.append(top_onehot)
                # print(results)
            
                # insert results into dictionary
                self.results_dict[val+"_sfmx"].update({image_name:float(sum(patch_summary_softmax)/25)})
                self.results_dict[val+"_top"].update({image_name:float(sum(patch_summary_top)/25)})

            # self.chondria_sfmx.update({image_name:float(sum(patch_summary_softmax)/25)})
            # self.chondria_top.update({image_name:float(sum(patch_summary_top)/25)})

            # print(self.chondria_sfmx)
            # print(self.chondria_top)
            # input()

        # add results to dataframe
        print("adding to dataframe")
        for key in self.class_dict.keys():
            val = self.class_dict[key]
            self.csv_data[val+'_sfmx'] = self.csv_data['REF'].map(self.results_dict[val+'_sfmx'])
            self.csv_data[val+'_top'] =  self.csv_data['REF'].map(self.results_dict[val+'_top'])
        # write dataframe to csv
        self.csv_data.to_csv(self.csv_out, index=False)
        print("DONE")

def main():
    args = arg_parse()
    rP = runPatches(args.source, args.save, args.shape, args.weights)
    rP.run()

if __name__=='__main__':
    main()
