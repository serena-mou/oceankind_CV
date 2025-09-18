# move select image/label pairs into another folder

#!/usr/bin/env python3

import shutil
import glob
import os
import re

all_im = glob.glob('/home/serena/Data/Fish/Raw/AUG25/images/*/*')
labels = glob.glob('/home/serena/Data/Fish/Raw/AUG25/labels/*/*.txt')

out = '/home/serena/Data/Fish/Processed/AUG25/ONLY_EPST/'

count_copy = 0
count = 0
print(len(all_im))
print(len(labels))
os.makedirs(os.path.join(out,"images"),exist_ok=True)
os.makedirs(os.path.join(out,"labels"),exist_ok=True)

# input()
for label in labels:
    label_name = label.split('/')[-1]
    f = open(label,"r")
    for line in f:
        line.rstrip()
        first_space = line.find(" ")
        cls = int(line[0:first_space]) 
    # classes of interest
    copy_cls = ["1", "2", "3", "4"]
    if str(cls) in copy_cls:
        #print("match: ",label)
        count_copy += 1
    else:
        continue  
    label_name_ex = label_name[:-4]
    # print(label_name)

    for i,im in enumerate(all_im):
        # match
        im = str(im)
        im_name = im.split('/')[-1]
        # print(str(im_name))
        period_idx = str(im_name).rfind(".")
        if period_idx != -1:
            im_name_ex = im_name[:int(period_idx)]
        else:
            print("ERROR BAD")

        # im_name = im_name_ex.split('.')[0]
        
        if label_name_ex == im_name_ex:    
        #if bool(re.match(label_name_ex,im_name_ex)) == True:
            im_out = os.path.join(out,"images",im_name)
            label_out = os.path.join(out,"labels",label_name)
            #print(im,label)
            #print(im_out, label_out) 
            shutil.copy(im,im_out)
            shutil.copy(label, label_out)
            count +=1
            #im_name == label_name:
            # print("found", im)
            #break
            # print(os.path.join(out,im_name_ex))
            #input()
            #shutil.copy(im,os.path.join(out,"images",im_name_ex))
        # if i == len(all_im)-1:
            # print("not found", label)
print("count: ",count)
print("count_copy: ", count_copy)