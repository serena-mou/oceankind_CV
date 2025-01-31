#!/usr/bin/env python3

"""
Author: Serena Mou
Created: 31 Jan 2024

===
Takes in all json files from cvat/COCO and generates a "master list" of classes as class_merger format yaml and as a master list
For use if list of classes don't match over numerous .json files
===

"""

import json
import os
import yaml
import glob
import csv
import sys
import argparse


class COCO2LIST():
    def __init__(self, json_file, save_location, new_cls):

        # load in the json file
        #json_file = "/home/serena/Data/SCTLD/RAW/1_100/annotations/instances_default.json" 
        #f = open(json_file)

        #self.data = json.load(f)
        self.in_files = json_file #"/home/serena/Data/SCTLD/RAW/"
        self.save_location = save_location #"/home/serena/Data/SCTLD/Processed/"
        self.new_cls = new_cls.split(",")
    def get_info(self, data):

        # given the json file, return lists of:
        # all the classes
        # all the image filenames
        # class of each annot (STARTS FROM 1) (length of number of annots)
        # image ids - a list associating each annotation with the image (length of number of annots)
        
        try:
            # List all the classes
            categories = data["categories"]
            classes = [category["name"] for category in categories]
            # print("all classes: ", len(classes), "\n")
            
            # List all the image filenames
            images = data["images"]
            img_names = [image["file_name"] for image in images]
            img_size = []
            for image in images:
                w = image["width"]
                h = image["height"]
                img_size.append([w,h])
            
            # For each annotation, get the class, image ID and the bbox
            annotations = data["annotations"]

            cls = [int(annotation["category_id"])-1 for annotation in annotations]
            img_ids = [int(annotation["image_id"])-1 for annotation in annotations]
            bbxs = [annotation["bbox"] for annotation in annotations]
            im_sz = [annotation["segmentation"]["size"] for annotation in annotations]
            # print(classes, img_names, cls, img_ids, bbxs, im_sz)
        except:
            print("ERROR: json file in wrong format - check it is downloaded from CVAT in COCO 1.0 format, from Segment Anything mask labels. ") 
        return classes, img_names, cls, img_ids, bbxs, im_sz
    
    def run(self):

        all_classes = {}
        all_in = []
        # check if the input is a single json file or a regex
        if os.path.isfile(self.in_files):
            all_in.append(self.in_files) # glob.glob(self.in_files+'Labels/*/*/*.json')
        else:
            try:
                all_in = glob.glob(self.in_files)
            except:
                print("ERROR: regex to multiple jsons failed")
        for i,data_path in enumerate(all_in):
            #print(all_in)
            # print(data_path)
            try:
                f = open(data_path)
                data = json.load(f)
            except:
                print("ERROR: json failed to load")
            # extract the info
            classes, img_names, cls, img_ids, bbxs, im_sz = self.get_info(data)

            for cls in classes:
                if cls not in all_classes.values():
                    if len(all_classes.keys()) is 0:
                        all_classes[0] = cls
                    else:
                        #print(all_classes.keys())
                        id = max(all_classes.keys())+1
                        all_classes[id] = cls

            #print(all_classes)
            #print(classes)
            #print(classes==prev_classes)
            #check_list = [e in classes for e in prev_classes]
            
            #print(classes)
            #print(check_list)
            #print(len(classes)
            new_cls_dict = {k:v for k,v in enumerate(self.new_cls)}
        merge = {
            "new_classes": new_cls_dict,
            "old_classes": all_classes
        }
         
        with open(os.path.join(self.save_location,"all_classes_merger.yaml"),"w") as outfile:
            yaml.dump(merge, outfile, sort_keys=False)
 
        with open(os.path.join(self.save_location,"all_classes_dict.yaml"),"w") as outfile:
            yaml.dump(all_classes, outfile, sort_keys=False)





def arg_parse():
    parser = argparse.ArgumentParser(description='Convert from COCO SAM annotation to YOLO format')

    parser.add_argument("--json", dest = "json_file",
            help = "Path to JSON file or regex to JSON files", default = None, type = str, required=True)
    
    parser.add_argument("--save", dest = "save_location",
            help = "Path to save labels", default = None, type = str, required=True)

    parser.add_argument("--newclasses", dest = "new_cls",
            help = "The class names to be merged to, separated by commas. Ex: cls0,cls1,cls2", 
            default = None, type = str, required=False)

    return parser.parse_args()


def main():
    args = arg_parse()

    #json_file = input("Path to JSON file or regex to files: ")
    #save_location = input("Path to save labels: ")
    test = COCO2LIST(args.json_file, args.save_location, args.new_cls)
    test.run()
    print("DONE")

if __name__=='__main__':
    main()
