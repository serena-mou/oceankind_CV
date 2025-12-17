'''
Iterate through images in a folder, check that the image has a matching label file. If there's no match, move the image. 
Written by: Serena Mou 
Date:       5/9/25
'''

import glob
import shutil
import os
import argparse


parser = argparse.ArgumentParser(description='Move images in folder that do not match any labels')
 
parser.add_argument("--image_folder", dest = "image_folder",
        help = "Path to folder of images", default = None, type = str)
parser.add_argument("--label_folder", dest = "label_folder",
        help = "Path to folder of labels", default = None, type = str)
parser.add_argument("--out_folder", dest = "out_folder",
        help = "Path to output folder", default = None, type = str)

args = parser.parse_args()

images = args.image_folder
labels = args.label_folder
copy_folder = args.out_folder


def get_name_from_path(path:str, end:bool):
        name = path.split('/')[-1]
        if end:
            return name
        base = name[0:name.rfind('.')]
        return base

match = False
all_ims = sorted(glob.glob(images))
all_labs = sorted(glob.glob(labels))
num_labels = len(all_labs)
print(len(all_ims))
for j,im in enumerate(all_ims):
    print(j)
    for i,lab in enumerate(all_labs):
        if get_name_from_path(path=im, end=False) == get_name_from_path(path=lab, end=False):
            # print("true")
            # input()
            match = True
            break
        if i == num_labels-1 and match == False:
            shutil.move(im, os.path.join(copy_folder, get_name_from_path(path=im, end=True)))
            print("NO MATCH: ", im)
        match = False
