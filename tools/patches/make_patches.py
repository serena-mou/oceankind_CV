# Take in folders of sorted images, crop em
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
        description="crop images"
    )

    parser.add_argument("-src","--source", dest="source",
        help = "Path to folder of images to crop", default=None, type=str, required=True
    )
    parser.add_argument("-s","--save", dest="save",
        help = "Path folder to save.", default=None, type=str, required=True
    )
    parser.add_argument("-sh","--shape", dest="shape",
        help = "Shape of output in x,y. Default 5x5", default="5,5", type=str, required=True
    )

    return parser.parse_args()



class makePatches():
    def __init__(self, src, save, shape):
        # init things
        self.all_ims = glob.glob(os.path.join(src,"*.jpg"))
        if len(self.all_ims) == 0:
            sys.exit("No .jpg files found in source. Check path.")

        if os.path.isdir(save): 
                print("Directory %s already exists. Continue? Y/N"%save)
                if input().lower() != 'y':
                    sys.exit("Exiting")
        os.makedirs(save, exist_ok=True)
        self.save = save

        self.shape = [int(x) for x in shape.split(',')]
    
    def run(self):
        # load images
        for im in self.all_ims:
            image = cv2.imread(im)
            image_name = os.path.basename(im)
            image_name = os.path.splitext(image_name)[0]
            # image size 
            u = int(image.shape[1]/self.shape[1])
            v = int(image.shape[0]/self.shape[0])

            for x in range(self.shape[1]):
                for y in range(self.shape[0]):
                    x_start, y_start, x_end, y_end = int(x*u), int(y*v), int((x+1)*u), int((y+1)*v)
                    cropped_im = image[y_start:y_end, x_start:x_end]
                    cv2.imwrite(os.path.join(self.save, image_name+"_%i_%i.jpg"%(x,y)), cropped_im)

        print("DONE")
def main():
    args = arg_parse()
    mP = makePatches(args.source, args.save, args.shape)
    mP.run()

if __name__=='__main__':
    main()
