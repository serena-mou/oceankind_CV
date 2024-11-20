#!/usr/bin/env python3

import glob
import os

sub_ims = glob.glob('/home/serena/Data/SCTLD/November/RAW/images/*')
all_labels = glob.glob('/home/serena/Data/SCTLD/November/Processed/brain_corals/all_labels/*.txt')

print(len(sub_ims))
print(len(all_labels))

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

print(sorted(not_im), len(not_im))
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
