#!/usr/bin/env python3

# dataset has a class I would like to delete

import glob
import csv

all_label_files = glob.glob("/home/serena/Data/xenia/Segmentation/Raw/Meiling_yolo/labels/train/*")
target_class = "1"


for label in all_label_files:
    # load file
    f = open(label, 'r')
    for line in f:
        #print(line)
        print(line.split(' ')[0].type)
        if line.split(' ')[0] == target_class:
            print(label)



#print(len(all_label_files))