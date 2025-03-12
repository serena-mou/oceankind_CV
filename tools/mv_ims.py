#!/usr/bin/env python3

import shutil
import glob
import os

all_im = glob.glob('/home/serena/Data/xenia/Raw/unzipped/*/images/*/*')
labels = glob.glob('/home/serena/Data/xenia/Processed/all_labels/*.txt')

out = '/home/serena/Data/xenia/Processed/all_images/'
print(len(all_im))
print(len(labels))

# input()
for label in labels:
    label_name = label.split('/')[-1]
    
    label_name = label_name.split('.')[0]
    # print(label_name)

    for i,im in enumerate(all_im):
        # match
        im = str(im)
        im_name_ex = im.split('/')[-1]
        im_name = im_name_ex.split('.')[0]
        
        
        if im_name == label_name:
            # print("found", im)
            #break
            # print(os.path.join(out,im_name_ex))
            #input()
            shutil.copy(im,os.path.join(out,im_name_ex))
        # if i == len(all_im)-1:
            # print("not found", label)
