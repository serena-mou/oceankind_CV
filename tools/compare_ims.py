#!/usr/bin/env python3

'''
Prints out all labels that don't match images
Written by: Serena Mou
Date:       17/9/25
'''

import glob
import os
import argparse

parser = argparse.ArgumentParser(description='Compare labels to images and print labels with no matching image')
 
parser.add_argument("--image_folder", dest = "image_folder",
        help = "Path to folder of images", default = None, type = str)
parser.add_argument("--label_folder", dest = "label_folder",
        help = "Path to folder of labels", default = None, type = str)
args = parser.parse_args()

sub_ims = glob.glob(os.path.join(args.image_folder,'*'))
all_labels = glob.glob(os.path.join(args.label_folder,'*.txt'))

print("Number of images: ",len(sub_ims))
print("Number of labels: ",len(all_labels))

label_list = []
im_list = []
for label in all_labels:
    label_name = label.split('/')[-1]
    
    label_name = label_name.split('.')[0]
    # print(label_name)
    label_list.append(label_name)

for i,im in enumerate(sub_ims):
    # match
    im = str(im)
    im_name_ex = im.split('/')[-1]
    im_name = im_name_ex.split('.')[0]
    im_list.append(im_name)

#print(label_list)
#print(im_list)
not_im = []

for label in label_list:
    if label not in im_list:
        not_im.append(label)


print("Labels with no image match:")
print(sorted(not_im))
print("Number of labels with no image match: ", len(not_im))
''' 
if im_name == label_name:
    print("found", im, label)
    break
    # print(os.path.join(out,im_name_ex))
    #input()
    #shutil.copy(im,os.path.join(out,im_name_ex))
elif i == len(sub_ims):
    print("not found",im, label)

'''
