'''
Script to find all label files with specified classes, remove those lines from the label file and save into a new folder. 
'''

#!/usr/bin/env python3


import glob
import csv
import os

all_label_files = glob.glob("/home/serena/Data/Fish/Processed/AUG25/ONLY_EPST/all_labels/*")
all_label_files_rm = "/home/serena/Data/Fish/Processed/AUG25/ONLY_EPST/all_labels_rm/"
os.makedirs(all_label_files_rm,exist_ok=True)
target_class_list = ['1','2','3','4']
#target_class_thres = 5
counter = 0

for label in all_label_files:
    label_name = label.split('/')[-1]
    # load file
    with open(label, 'r') as f:
        lines = f.readlines()
    '''
        for line in lines:
            if int(line.split(' ')[0]) >= target_class_thres:
                print(label_name)
    '''    
    with open(os.path.join(all_label_files_rm, label_name),'w') as g:
        for line in lines:
        #print(line)
        #print(line.split(' ')[0].type)
            if int(line.split(' ')[0]) in target_class_list:
                g.write(line)
                #print(label)
            else:
                counter +=1
print("number of lines removed: ",counter)



#print(len(all_label_files))