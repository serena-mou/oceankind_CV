#/usr/bin/env python3

"""
Author: Serena Mou
Created: 23 July 2024

===
Converts the Segment Anything Model (SAM) masks from CVAT in COCO format into YOLO compatible bounding boxes
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
    def __init__(self, json_file, save_location, merger, merge_file, new_cls):

        # load in the json file
        #json_file = "/home/serena/Data/SCTLD/RAW/1_100/annotations/instances_default.json" 
        #f = open(json_file)

        #self.data = json.load(f)
        self.in_files = json_file #"/home/serena/Data/SCTLD/RAW/"
        self.save_location = save_location #"/home/serena/Data/SCTLD/Processed/"
        self.merger = merger
        self.merge_file = merge_file
        self.new_cls = new_cls


    def get_info(self, data):

        # given the json file, return lists of:
        # all the classes
        # all the image filenames
        # class of each annot (STARTS FROM 1) (length of number of annots)
        # image ids - a list associating each annotation with the image (length of number of annots)
        

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
        
        return classes, img_names, cls, img_ids, bbxs, im_sz
    



    def write_yaml(self, classes):
        # generate yaml file with each class name
        yaml_path = os.path.join(self.save_location,"data.yaml")
        test_yaml_path = os.path.join(self.save_location,"test.yaml")
        # dictionary of {0: class0, 1: class1...}
        cls_dict = {k:v for k,v in enumerate(classes)}

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
  
    def write_class_merger(self, classes):
        # generate yaml file with new classes and old classes
        # for the purpose of merging classes 
        yaml_path = os.path.join(self.save_location,"class_merger.yaml")

        # dictionary of {0: class0, 1: class1...}
        cls_dict = {k:v for k,v in enumerate(classes)}
        new_cls = self.new_cls.split(',')
        
        new_cls_dict =  {k:v for k,v in enumerate(new_cls)}

        data = {
            "new_classes": new_cls_dict,
            "old_classes": cls_dict
        }

        with open(yaml_path, 'w') as outfile:
            yaml.dump(data, outfile, sort_keys=False)

    
    def write_label_summary(self, summary_dict):
        csv_path = os.path.join(self.save_location,"label_summary.csv")


        with open(csv_path, 'w') as outfile:
            writer = csv.writer(outfile)
            for key, value in summary_dict.items():
                writer.writerow([key,value])


    def read_class_merger(self, merger_yaml):
        # reading yaml file with ,new_cls_idx on end of each line 
        # return dictionary of mappings to new classes


        new_class_mapping = {}
        with open(merger_yaml, 'r') as stream:
            data_loaded = yaml.safe_load(stream)

        to_merge = data_loaded['old_classes']
        new_classes = data_loaded['new_classes']
        
        self.new_cls = []
        for cls in new_classes:
            self.new_cls.append(new_classes[cls])
        
        for cls in to_merge:
            if ',' not in str(to_merge[cls]):
                sys.exit("class merger yaml not updated to correct format")
            [n, merge] = str(to_merge[cls]).split(',')
            if int(merge) not in new_class_mapping:
                new_class_mapping[int(merge)] = [cls]
            else:
                new_class_mapping[int(merge)].append(cls)

        return new_class_mapping


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

    def write_txt(self, classes, img_names, cls, img_ids, bbxs, im_sz, mapping):
        
        # location to save labels
        out_folder = os.path.join(self.save_location,"all_labels")

        if not os.path.isdir(out_folder):
            os.mkdir(out_folder)

        # for each image, write a textfile of name image_name.txt
        # for each annotation in the image, write
        # class, bb centre x, bb centre y, bb w, bb h 

    
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
                # print(idx, cls, mapping)
                # if classes are being remapped then use the mapping (dict) to find new class
                if mapping is not None:
                    self.write_yaml(self.new_cls)
                    for new_cls in mapping:
                        if cls[idx] in mapping[new_cls]:
                            idx_class = new_cls
                        else:
                            continue# print("Remapping of classes error")

                else: 
                    idx_class = cls[idx] 
                    #print(cls,idx)
                    self.write_yaml(classes)
                [xn, yn, wn, hn] = self.bbx_converter(bbxs[idx], im_sz[idx])
                lines.append((idx_class, xn, yn, wn, hn))

            
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
        for i in classes:
            if i not in summary_dict:
                summary_dict[i] = 0

        for i, cls_idx in enumerate(cls): 
            summary_dict[classes[cls_idx]] +=1

            # if classes[cls_idx-1] == "remove_Orbicellla_y_SCTLD":
            #         print(img_names[img_ids[i]-1])

        #print(cls)
        #input()
    
    def run(self):

        all_in = []
        if os.path.isfile(self.in_files):
            all_in.append(self.in_files) # glob.glob(self.in_files+'Labels/*/*/*.json')
        else:
            all_in = glob.glob(self.in_files)
        
        summary_dict = {}
        # For each json file
        for data_path in all_in:
            # print(data_path)

            f = open(data_path)
            data = json.load(f)
            # extract the info
            classes, img_names, cls, img_ids, bbxs, im_sz = self.get_info(data)
            # print(classes)
            # print(img_names)
            # input()
            ## For the first run, generate the class merger file            
            if self.merger:
                if self.merge_file:
                    mapping = self.read_class_merger(self.merge_file)

                elif self.merge_file == None:
                    self.write_class_merger(classes)
                    print("\n**NOTE** An assumption is being made that if multiple jsons are being loaded, all classes exist in first json to be loaded\n")
                    sys.exit("Class merger file written to %s, please update and load on next run. Exiting... "%(os.path.join(self.save_location,"class_merger.yaml")))

            else:
                mapping = None    
            
            self.write_txt(classes, img_names, cls, img_ids, bbxs, im_sz, mapping)

            # self.write_txt(img_names, cls, img_ids, bbxs, im_sz, mapping)
            
            ## Label summary
            # self.label_summary(classes, img_names, img_ids, cls, summary_dict)
            # self.write_label_summary(summary_dict) 
            
            ## Use this section to move the images that are referenced in the jsons 
            # for img_name in img_names:
            #     src = os.path.join(images_path,img_name)
            #     if os.path.isfile(src):
            #         dest = os.path.join(self.save_location,"training/all_images",img_name)
            #         shutil.copy2(src,dest)
            #     else: print("not a file", src)


def arg_parse():
    parser = argparse.ArgumentParser(description='Convert from COCO SAM annotation to YOLO format')

    parser.add_argument("--json", dest = "json_file",
            help = "Path to JSON file or regex to JSON files", default = None, type = str, required=True)
    
    parser.add_argument("--save", dest = "save_location",
            help = "Path to save labels", default = None, type = str, required=True)
    
    parser.add_argument("--merge", dest = "merge_file",
            help = "Path to merge yaml file", default = None, type = str, required=False)
    
    parser.add_argument("--newclasses", dest = "new_cls",
            help = "The class names to be merged to, separated by commas. Ex: cls0,cls1,cls2", default = None, type = str, required=False)

    return parser.parse_args()


def main():
    args = arg_parse()

    #json_file = input("Path to JSON file or regex to files: ")
    #save_location = input("Path to save labels: ")
    merge = input("Do you need to merge classes? Y/N: ")
    if merge == "Y":
        merger = True

        ex_merge_file = input("\nDo you want to use an existing merge file? Y/N: ")
        if ex_merge_file == "Y":
            if args.merge_file == None:
                sys.exit("Please add merge file path argument to --merge")
            
            merge_file = args.merge_file#input("Path to merge yaml: ")
            if args.new_cls:
                print("--newclasses being ignored\n")
            new_cls = None
        elif ex_merge_file == "N":
            if args.merge_file:
                print("--merge being ignored\n")
            merge_file = None
            if args.new_cls == None:
                sys.exit("Please add new classes argument to --newclasses")
            new_cls = args.new_cls#input("What are the new classes to merge to? cls1,cls2,...,clsn : ")
        else:
            raise ValueError("Please type Y or N and enter")

    elif merge == "N":
        if args.merge_file:
            print("--merge being ignored\n")
        if args.new_cls:
            print("--newclasses being ignored\n")
        merger = False
        merge_file = None
        new_cls = None
    else:
        raise ValueError("Please type Y or N and enter")

    test = COCO2YOLOBB(args.json_file, args.save_location, merger, merge_file, new_cls)
    test.run()
    print("DONE")

if __name__=='__main__':
    main()