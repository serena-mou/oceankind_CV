#!/usr/bin/env python3

"""
Author: Serena Mou
Created: 14 May 2026

===
Converts yolo bounding box labels into cropped images in folders with class names
===

"""

import glob
import yaml
import argparse
import os
import sys
import cv2

class bb2class():
    def __init__(self, root, save):
        self.all_im_path = os.path.join(root,"all_images")
        self.all_labs = glob.glob(os.path.join(root,"all_labels","*"))
        self.data = os.path.join(root,"data.yaml")
        self.classes = {}
        # if save is not recorded, save in root folder
        if save is None:
            self.save_path = root
        else: 
            self.save_path = save

        # load yaml to get classes
        with open(self.data, 'r') as f:
            self.classes = yaml.load(f, Loader=yaml.SafeLoader) 
        self.classes = self.classes['names']

        # make folders for each class
        for val in self.classes.values():
            os.makedirs(os.path.join(self.save_path,val),exist_ok=True)

    def bb2corners(self, bbx, imsz):
        # convert from yolo bounding box format (normalised) [xmiddle, ymiddle, width, height] to 
        # corners (pixels) [topleft x, topleft y, bottomright x, bottomright y]
        [xn, yn, wn, hn] = bbx
        (fh, fw) = imsz
        
        # convert from normalised x,y,w,h to pixel values
        xp = xn*fw
        yp = yn*fh
        wp = wn*fw
        hp = hn*fh

        tlx = int(xp-(wp/2))
        tly = int(yp-(hp/2))
        brx = int(xp+(wp/2))
        bry = int(yp+(hp/2))

        return [tlx, tly, brx, bry]


    def run(self):
        # for each label file, if there are annotations, load image, create crop, save 
        for lab in self.all_labs:
            if os.stat(lab).st_size > 0:
                im_name = os.path.basename(lab)
                im_name = im_name.rsplit(".",1)[0]
                im_loc = glob.glob(os.path.join(self.all_im_path,im_name+'*'))
                if len(im_loc) > 1:
                    sys.exit("Label name (%s) matched with more than one image (%s)"%(lab,im_loc))
                image = cv2.imread(im_loc[0])
                f = open(lab,'r')
                count=0
                for line in f:
                    cls = int(line[0:line.find(" ")])
                    bb = line[line.find(" ")+1:-1]
                    # bb =bb.split(' ')
                    bb = [float(x) for x in bb.split(" ")]
                    imsz = image.shape[0:2]
                    
                    [tlx, tly, brx, bry] = self.bb2corners(bb,imsz)
                    im_crop = image[tly:bry, tlx:brx]
                    cv2.imwrite(os.path.join(self.save_path,self.classes[cls],"%s-%i.jpg"%(im_name,count)),im_crop) 
                    count+=1
                    
                    # im_smol = cv2.resize(image, (0,0), fx=0.1, fy=0.1)
                    # cv2.imshow("crop", im_crop)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()



def arg_parse():
    parser = argparse.ArgumentParser(description='Convert from Bounding Boxes to cropped images for classifier')

    parser.add_argument("--root", dest = "root",
            help = "Path to folder with all_images and all_labels folders and data.yaml in yolo format", default = None, type = str, required=True)
    
    parser.add_argument("--save", dest = "save",
            help = "Path to save cropped images", default = None, type = str, required=False)
    
    return parser.parse_args()


def main():
    args = arg_parse()

    #json_file = input("Path to JSON file or regex to files: ")
    #save_location = input("Path to save labels: ")
    cvt = bb2class(args.root, args.save)
    cvt.run()
    print("DONE")

if __name__=='__main__':
    main()


