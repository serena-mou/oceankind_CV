#!/usr/bin/env python3

"""
Author: Matthew Tavatgis, Updated by SM
Created: 27 Jan 2024

Creates a train / test / validation split in the format expected by yolov5+
Optionally trim n images with empty labels
"""

import os
import sys
import shutil
import random
import argparse
import numpy as np
from statistics import mode
from sklearn.model_selection import StratifiedShuffleSplit
import glob

random.seed(1)

def arg_parse():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Create train test split for yolov5+')

    parser.add_argument("--src", dest = "src_dir",
            help = "Source directory to parse", default = None, type = str)
    parser.add_argument("--valid", dest = "valid",
            help = "Fraction to split for validation, 0-1", default = 0.2, type = float)
    parser.add_argument("--test", dest = "test",
            help = "Fraction to split for testing, 0-1", default = None, type = float)
    parser.add_argument("--dump", dest = "n_dump",
            help = "Number of empty images to drop", default = None, type = int)
    parser.add_argument("--rand", dest = "random_state",
            help = "Seed for random generation", default = 1, type = int)

    return parser.parse_args()

def main():
    """
    Creates new directory under source and randomly coppies specified percentage of images and labels to 'train' or 'valid'
    """
    # Get arguments
    args = arg_parse()

    # Check source images exist
    image_source_dir = os.path.join(args.src_dir, "all_images")
    if os.path.exists(image_source_dir) == False:
        raise FileNotFoundError("Source directory {} does not exist".format(image_source_dir))

    # Check source labels exist
    label_source_dir = os.path.join(args.src_dir, "all_labels")
    if os.path.exists(image_source_dir) == False:
        raise FileNotFoundError("Source directory {} does not exist".format(label_source_dir))
       # Check labels and images match - if an image has no label file, generate an empty one. If a label has no image, throw a warning
    all_images = glob.glob(os.path.join(image_source_dir,"*"))
    all_labels = glob.glob(os.path.join(label_source_dir,"*"))

    # remove extensions
    all_ims = []
    all_labs = []
 
    for img in all_images:
        img_name = img.split('/')[-1]
        idx = img_name.rfind('.')
        img_name = img_name[0:idx]
        all_ims.append(img_name)


    for label in all_labels:
        label_name = label.split('/')[-1]
        idx = label_name.rfind('.') 
        label_name = label_name[0:idx]
        all_labs.append(label_name)      


    # all_ims = [im.split("/")[-1].split('.')[0] for im in all_ims]
    # all_labs = [lab.split("/")[-1].split('.')[0] for lab in all_labs]
    # print(all_ims)
    # print(all_labs)
    for im in all_ims:
        if im not in all_labs:
            print("Label file does not exist for %s, creating empty label"%im)
            empty_label = os.path.join(label_source_dir,im+".txt")
            open(empty_label, 'a').close()
    
    for lab in all_labs:
        if lab not in all_ims:
            print("WARNING: Label file %s exists with no matching image..."%lab) 
    # Create output directory paths
    train_out_dir = os.path.join(args.src_dir, "train")
    valid_out_dir = os.path.join(args.src_dir, "valid")
    test_out_dir = os.path.join(args.src_dir, "test")
    train_image_dir = os.path.join(train_out_dir, "images")
    train_label_dir = os.path.join(train_out_dir, "labels")
    valid_image_dir = os.path.join(valid_out_dir, "images")
    valid_label_dir = os.path.join(valid_out_dir, "labels")
    test_image_dir = os.path.join(test_out_dir, "images")
    test_label_dir = os.path.join(test_out_dir, "labels")

    # Prompt to overwrite if output dir exists, otherwise attempt to create it
    if os.path.exists(train_out_dir) or os.path.exists(valid_out_dir) or os.path.exists(test_out_dir):
        print("WARNING: Output Directory Exists, data will be overwritten ", end="")

        if input("Y/N?:").lower() != "y":
            print("EXITING...\n")
            exit()
        else:
            print("CONTINUING...\n")
            # Delete existing output dir
            try:
                if os.path.exists(train_out_dir): shutil.rmtree(train_out_dir)
                if os.path.exists(valid_out_dir): shutil.rmtree(valid_out_dir)
                if os.path.exists(test_out_dir): shutil.rmtree(test_out_dir)
            except OSError as error:
                print(error)
                sys.exit()

    print("#################### Creating output directories ####################")
    print("TRAIN: {}\nVALID: {}".format(train_out_dir, valid_out_dir))

    try:
        os.mkdir(train_out_dir)
        os.mkdir(valid_out_dir)
        os.mkdir(train_image_dir)
        os.mkdir(train_label_dir)
        os.mkdir(valid_image_dir)
        os.mkdir(valid_label_dir)
    except OSError as error:
        print(error)
        sys.exit()

    if args.test is not None:
        print("TEST: {}\n".format(test_out_dir))
        try:
            os.mkdir(test_out_dir)
            os.mkdir(test_image_dir)
            os.mkdir(test_label_dir)
        except OSError as error:
            print(error)
            sys.exit()
    else:
        print("")

    # Get image paths
    images = os.listdir(image_source_dir)
    images.sort()

    # Get label paths
    labels = os.listdir(label_source_dir)
    labels.sort()
    
    # If dumping empty images
    if args.n_dump is not None:
        
        # Parse labels looking for empty sets to dump
        empty = []
        for i, label in enumerate(labels):
            if os.stat(os.path.join(label_source_dir, label)).st_size == 0:
                empty.append(i)
        
        # Check numbers
        if args.n_dump > len(empty):
            print("ERROR: Number of empty dumps requested is greater than number of empty labels! EXITING...\n")
            sys.exit()
        
        # Get a random selection of targets to remove
        targets = random.sample(empty, args.n_dump)

        # Purge targets from inputs
        for i in sorted(targets, reverse=True):
            del images[i]
            del labels[i]

    # Get label mode for each image
    label_modes = []
    for label_name in labels:

        # Get path
        label_path = os.path.join(label_source_dir, label_name)
        
        # Check if empty
        if os.stat(label_path).st_size != 0:
            
            # If not empty get label mode
            with open(label_path, 'r') as stream:
                lines = stream.readlines()
                
                # Iterate label lines
                cls_list = []
                for label in lines:
                    label = label.rstrip()[0]
                    cls_list.append(label)
                
                # Get mode
                label_modes.append(mode(cls_list))
        else:
            # If empty use -1 place holder
            label_modes.append(-1)

    # Get train test splits
    if args.test is None:
        # Split train and valdidation sets using stratified (balance preserving split)
        train_images, valid_images, train_labels, valid_labels = set_split(images, labels, label_modes, args.valid)[:4]

    else:
        # Calculate split percentages
        total_per = args.valid + args.test
        test_per = args.test / total_per
        
        # Split train, valdidation and test sets using stratified (balance preserving split)
        train_images, temp_images, train_labels, temp_labels, temp_modes = set_split(images, labels, label_modes, total_per)
        valid_images, test_images, valid_labels, test_labels = set_split(temp_images, temp_labels, temp_modes, test_per)[:4]
    
    # Copy splits to output folders
    # Train images
    for img_name in train_images:
        # Make paths
        temp_src_dir = os.path.join(image_source_dir, img_name)
        temp_dst_dir = os.path.join(train_image_dir, img_name)

        # Copy
        copy_file(temp_src_dir, temp_dst_dir)
        
    # Train labels
    for label_name in train_labels:
        # Make paths
        temp_src_dir = os.path.join(label_source_dir, label_name)
        temp_dst_dir = os.path.join(train_label_dir, label_name)

        # Copy
        copy_file(temp_src_dir, temp_dst_dir)
    
    # Valid images
    for img_name in valid_images:
        # Make paths
        temp_src_dir = os.path.join(image_source_dir, img_name)
        temp_dst_dir = os.path.join(valid_image_dir, img_name)

        # Copy
        copy_file(temp_src_dir, temp_dst_dir)
        
    # Valid labels
    for label_name in valid_labels:
        # Make paths
        temp_src_dir = os.path.join(label_source_dir, label_name)
        temp_dst_dir = os.path.join(valid_label_dir, label_name)

        # Copy
        copy_file(temp_src_dir, temp_dst_dir)
    
    # Test set
    if args.test is not None:
        # Test images
        for img_name in test_images:
            # Make paths
            temp_src_dir = os.path.join(image_source_dir, img_name)
            temp_dst_dir = os.path.join(test_image_dir, img_name)

            # Copy
            copy_file(temp_src_dir, temp_dst_dir)
            
        # Test labels
        for label_name in test_labels:
            # Make paths
            temp_src_dir = os.path.join(label_source_dir, label_name)
            temp_dst_dir = os.path.join(test_label_dir, label_name)

            # Copy
            copy_file(temp_src_dir, temp_dst_dir)

# Helper function for train test splits
def set_split(src_images, src_labels, src_label_idents, split):
    
    # Generate splits
    train_split = StratifiedShuffleSplit(n_splits=1, test_size = split, random_state = 1)
    train_gen = train_split.split(np.zeros(len(src_label_idents)), src_label_idents)
    train_index, valid_index = next(train_gen)
    
    # Output arrays
    train_images = []
    train_labels = []
    valid_images = []
    valid_labels = []
    valid_idents = []
    
    # Fill 
    for i in train_index:
        train_images.append(src_images[i])
        train_labels.append(src_labels[i])

    for i in valid_index:
        valid_images.append(src_images[i])
        valid_labels.append(src_labels[i])
        valid_idents.append(src_label_idents[i])

    return train_images, valid_images, train_labels, valid_labels, valid_idents

# Helper function to copy files
def copy_file(src_dir, dst_dir):
    try:
        shutil.copy(src_dir, dst_dir)
    except OSError as error:
        print(error)
        print("\nFailed to copy {} to {}".format(src_dir, dst_dir))
        sys.exit()

if __name__=="__main__":
    main()
