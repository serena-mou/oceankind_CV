# sort a folder of images into different folders
# for patch classification, sort images with entire coverage first
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

def arg_parse():

    parser = argparse.ArgumentParser(
        description="sort images into folders according to keypress"
    )

    parser.add_argument("-src","--source", dest="source",
        help = "Path to folder of images to sort", default=None, type=str, required=True
    )
    parser.add_argument("-o","--output", dest="output",
        help = "Name of folders to sort into, separated by commas. Eg. Chondria,Background,Skip", default=None, type=str, required=True
    )
    parser.add_argument("-k","--key", dest="key",
        help = "Key press to sort folder into. Must match save order as --output. Eg. C,B,S", default=None, type=str, required=True
    )
    parser.add_argument("-s","--scale", dest="scale",
        help = "Scale image size for viewing. Default 0.2", default=0.2, type=float, required=False
    )



    return parser.parse_args()


class sortData():
    def __init__(self, src, out, key, scale):
        # parse the args and check they're valid
        
        
        # get all in the images in source folder
        ## TODO add other image file extentions

        self.all_ims = glob.glob(os.path.join(src,"*.jpg"))
        if len(self.all_ims) == 0:
            sys.exit("No .jpg files found in source. Check path.")
        
        self.output = out.split(',')
        self.keys = key.split(',')

        if len(self.output) != len(self.keys):
            sys.exit("Number of output folders and number of keys have to match.")

        unique = list(dict.fromkeys(self.keys))
        if len(self.keys) != len(unique):
            sys.exit("Keys need to be unique")

        self.scale = scale

        # convert keys to lowered ord
        self.keys_ord = [ord(x.lower()) for x in self.keys]

        # make dirs
        self.output = [os.path.join(src,x) for x in self.output]
        for o in self.output:
            if os.path.isdir(o): 
                print("Directory %s already exists. Continue? Y/N"%o)
                if input().lower() != 'y':
                    sys.exit("Exiting")
            os.makedirs(o, exist_ok=True)
        
    def run(self):
        # create folders in source folder
        for im in self.all_ims:
            # load resized cv2 image
            image = cv2.resize(cv2.imread(im),(0,0), fx=self.scale, fy=self.scale)
            cv2.imshow('image', image)

            while 1:
                k = cv2.waitKey(1) 
                #print(k)
                if k in self.keys_ord:
                    idx = self.keys_ord.index(k)
                    shutil.move(im,self.output[idx])
                    break

            cv2.destroyAllWindows()
            # switch = 0
                

def main():
    args = arg_parse()
    sD = sortData(args.source, args.output, args.key, args.scale)
    sD.run()

if __name__=='__main__':
    main()

