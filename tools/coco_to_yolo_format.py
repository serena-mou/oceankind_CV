#!/usr/bin/env python3

"""
Author: Serena Mou
Created: 23 July 2024

===
Converts the Segment Anything Model (SAM) masks from CVAT in COCO format into YOLO compatible bounding boxes
If there are multiple jsons and the classes do not all match, a "all_classes_dict.yaml" will be required
===

"""

import json
import os
import yaml
import glob
import csv
import sys
import argparse


class COCO2YOLOBB():
    def __init__(self, json_file, save_location, classes_dict):

        # load in the json file
        #json_file = "/home/serena/Data/SCTLD/RAW/1_100/annotations/instances_default.json" 
        #f = open(json_file)

        #self.data = json.load(f)
        self.in_files = json_file #"/home/serena/Data/SCTLD/RAW/"
        self.save_location = save_location #"/home/serena/Data/SCTLD/Processed/"
        self.all_classes_dict = {}
        if classes_dict is not None: 
            with open(classes_dict, 'r') as f:
                self.all_classes_dict = yaml.load(f, Loader=yaml.SafeLoader) 

    def get_info(self, data):

        # given the json file, return lists of:
        # all the classes as a dict
        # all the image filenames
        # class of each annot (STARTS FROM 1) (length of number of annots)
        # image ids - a list associating each annotation with the image (length of number of annots)
        try:
            # List all the classes
            categories = data["categories"]
            classes = [category["name"] for category in categories]
            classes_dict = {k:v for k,v in enumerate(classes)}
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
            return classes_dict, img_names, cls, img_ids, bbxs, im_sz

            
        except:
            print("ERROR: json file in wrong format - check it is downloaded from CVAT in COCO 1.0 format, from Segment Anything mask labels. ") 
    



    def write_yaml(self, classes):
        # generate yaml file with each class name
        yaml_path = os.path.join(self.save_location,"data.yaml")
        test_yaml_path = os.path.join(self.save_location,"test.yaml")

        if os.path.isfile(yaml_path):
            ow = input("data.yaml already exists at %s. Overwrite? Y/N "%(yaml_path))
            if ow.lower() == "y":
                print("Overwriting data.yaml")
            else:
                sys.exit("ERROR: Not overwriting data.yaml. Please select different save path without exisiting yaml file.")
        
        if os.path.isfile(test_yaml_path):
            ow1 = input("test.yaml already exists at %s. Overwrite? Y/N "%(yaml_path))
            if ow.lower() == "y":
                print("Overwriting test.yaml")
            else:
                sys.exit("ERROR: Not overwriting test.yaml. Please select different save path without exisiting yaml file.")
  

        # dictionary of {0: class0, 1: class1...}
        if len(self.all_classes_dict.keys()) > 0:
            cls_dict = self.all_classes_dict
        else:
            cls_dict = classes #{k:v for k,v in enumerate(classes)}

        data = {
            "path":self.save_location,
            "train": "train",
            "val": "valid",
            "names": cls_dict
        }

        test = {
            "path":self.save_location,
            "train": "train",
            "val": "test",
            "names": cls_dict
        }


        with open(yaml_path, 'w') as outfile:
            yaml.dump(data, outfile, sort_keys=False)

        with open(test_yaml_path, 'w') as outfile:
            yaml.dump(test, outfile, sort_keys=False)
  
    def write_label_summary(self, summary_dict):
        csv_path = os.path.join(self.save_location,"label_summary.csv")


        with open(csv_path, 'w') as outfile:
            writer = csv.writer(outfile)
            for key, value in summary_dict.items():
                writer.writerow([key,value])


    def bbx_converter(self, bbx_raw, im_sz):
        # given a bounding box in pixel format [x top left, y top left, width, height]
        # return in yolo format in normalised to image size [x middle, y middle, width, height]
        [xl, yl, w, h] = bbx_raw
        [fh, fw] = im_sz
        xn = (xl + (w/2))/fw
        yn = (yl + (h/2))/fh
        wn = (w/fw)
        hn = (h/fh)

        return [xn, yn, wn, hn]

    def write_txt(self, classes, img_names, cls, img_ids, bbxs, im_sz, loop):
        
        # location to save labels
        out_folder = os.path.join(self.save_location,"all_labels")
        if loop == 0:
            if not os.path.isdir(out_folder):
                os.mkdir(out_folder)
            else:
                ow = input("WARNING: Folder %s already exists. Overwrite contents? Y/N "%(out_folder))
                if ow.lower() == "y":
                    print("Overwriting labels")
                else:
                    sys.exit("ERROR: Not overwriting, use different path in --save argument")

            # for each image, write a textfile of name image_name.txt
            # for each annotation in the image, write
            # class, bb centre x, bb centre y, bb w, bb h 
            
            # write data.yaml and test.yaml
            self.write_yaml(classes)

        
        for i, name in enumerate(img_names):
            # textfile name 
            get_name_str = name.split('.')
            if len(get_name_str)>2:
                print("Image name not able to be processed")
                print(name)
            else:
                out_txt_name = get_name_str[0]+'.txt'
                # print(out_txt_name)

            # get the indexes of all annotations in this image
            all_im_idx = [j for j in range(len(img_ids)) if img_ids[j] == i]
            # print(img_ids)
            # print(cls)
            # print(out_txt_name, all_im_idx)
            
            # list of lines for each text file consisting of class, x, y, w, h (normalised)
            lines = []

            for idx in all_im_idx:
                #print(idx)
                idx_class = cls[idx] 
                #print(idx_class)
                #print(classes)
                #print(self.all_classes_dict)
                if len(self.all_classes_dict.keys()) < 1:
                    full_list_class = idx_class 
                else:
                    full_list_class = list(self.all_classes_dict.keys())[list(self.all_classes_dict.values()).index(classes[idx_class])]
                #input()
                [xn, yn, wn, hn] = self.bbx_converter(bbxs[idx], im_sz[idx])
                lines.append((full_list_class, xn, yn, wn, hn))

            
            out_path = os.path.join(out_folder,out_txt_name) 

            # if os.path.isfile(out_path):
            #     print("FILE %s ALREADY EXISTS. CONTINUE?" % out_path)
            #     input()

            # write to file
            with open(out_path,'w') as f:
                for line in lines:
                    write_line = "%d %0.4f %0.4f %0.4f %0.4f"%line
                    # print(write_line)
                    f.write("%s\n"%write_line)
        
            
                    

    def label_summary(self, classes, img_names, img_ids, cls, summary_dict):
        #print(classes, img_names, cls)
        for i in classes.values():
            if i not in summary_dict:
                summary_dict[i] = 0
        #print(summary_dict)
    #input()

        for i, cls_idx in enumerate(cls): 
            summary_dict[classes[cls_idx]] +=1

            # if classes[cls_idx-1] == "remove_Orbicellla_y_SCTLD":
            #         print(img_names[img_ids[i]-1])
        return summary_dict
        #print(cls)
        #input()
    
    def run(self):


        all_in = []
        # check if the input is a single json file or a regex
        if os.path.isfile(self.in_files):
            all_in.append(self.in_files) # glob.glob(self.in_files+'Labels/*/*/*.json')
        else:
            try:
                all_in = glob.glob(self.in_files)
            except:
                print("ERROR: regex to multiple jsons failed")


        summary_dict = {}
        # For each json file
        for i,data_path in enumerate(all_in):
            #print(all_in)
            # print(data_path)
            try:
                f = open(data_path)
                data = json.load(f)
            except:
                print("ERROR: json failed to load")
            # extract the infoi
            #print(self.classes_dict)
            classes, img_names, cls, img_ids, bbxs, im_sz = self.get_info(data)
            #print(classes)
            #input()
            # print(classes)
            # print(img_names)
            # input()
            ## For the first run, generate the class merger file            

            self.write_txt(classes, img_names, cls, img_ids, bbxs, im_sz, i)

            # self.write_txt(img_names, cls, img_ids, bbxs, im_sz, mapping)
            
            ## Label summary
            #summary_dict = self.label_summary(classes, img_names, img_ids, cls, summary_dict)
        
        #self.write_label_summary(summary_dict) 
            
            ## Use this section to move the images that are referenced in the jsons 
            # for img_name in img_names:
            #     src = os.path.join(images_path,img_name)
            #     if os.path.isfile(src):
            #         dest = os.path.join(self.save_location,"training/all_images",img_name)
            #         shutil.copy2(src,dest)
            #     else: print("not a file", src)
        
        # print(summary_dict)

def arg_parse():
    parser = argparse.ArgumentParser(description='Convert from COCO SAM annotation to YOLO format')

    parser.add_argument("--json", dest = "json_file",
            help = "Path to JSON file or regex to JSON files", default = None, type = str, required=True)
    
    parser.add_argument("--save", dest = "save_location",
            help = "Path to save labels", default = None, type = str, required=True)
    
    parser.add_argument("--classes", dest = "classes_dict",
            help = "Path to all_classes yaml file", 
            default = None, type = str, required=False)
    return parser.parse_args()


def main():
    args = arg_parse()

    #json_file = input("Path to JSON file or regex to files: ")
    #save_location = input("Path to save labels: ")
    test = COCO2YOLOBB(args.json_file, args.save_location, args.classes_dict)
    test.run()
    print("DONE")

if __name__=='__main__':
    main()
