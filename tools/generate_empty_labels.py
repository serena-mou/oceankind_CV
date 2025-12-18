'''
Take in a label folder and image folder, generate empty label files for all images with no label file
(Assuming these images have no objects in them)


Written by: Serena Mou 
Date:       16/12/25
'''


import glob
import os
import argparse
import shutil

def arg_parse():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Generate empty label files for ims with no labels')

    parser.add_argument("--img_folder", dest = "img_folder",
            help = "Path to folder of images", default = "all_images", type = str)
    parser.add_argument("--label_folder", dest = "label_folder",
            help = "Path of folder of labels", default = "all_labels", type = str)


    return parser.parse_args()



def main():
    args = arg_parse()

    all_images = glob.glob(os.path.join(args.img_folder,'*'))
    all_labels = glob.glob(os.path.join(args.label_folder,'*'))
       
           
    label_list = []
    for label in all_labels:
        label_name = label.split('/')[-1]
        idx = label_name.rfind('.') 
        label_name = label_name[0:idx]
        # print(label_name)
        label_list.append(label_name)      
        # get matching image
    
    # For each image
    for img in all_images:
        # check if there's a matching label
        # image name
        img_name = img.split('/')[-1]
        dot_idx = img_name.rfind('.')
        img_name = img_name[0:dot_idx]
        # if not
        # full label path
        if img_name not in label_list:
            empty_label = os.path.join(args.label_folder, img_name+".txt")
            open(empty_label, 'a').close()
            continue
if __name__=='__main__':
    main()
