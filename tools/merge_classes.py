#!/usr/bin/env python3

"""
Author: Serena Mou
Created:  16 October 2024

===
Merges classes with a class_merger.yaml for labels that are in YOLO format
===

"""

import os
import yaml
import glob
import sys
import argparse

class mergeClasses():
    def __init__(self, labels_in, save, merge_file, data_file, new_cls, use_case):

        self.labels_in = labels_in
        self.save = save
        self.merge_file = merge_file
        self.data_file = data_file
        # extract classes from data.yaml
        if self.data_file is not None:
            with open(self.data_file,'r') as stream:
                data_loaded = yaml.safe_load(stream)

            try:
                self.old_cls_dict = data_loaded["names"]
            except:
                sys.exit("--data has no \"names\" section of original class names")
        if new_cls is not None:
            try:
                new_cls = new_cls.split(',')
                self.new_cls_dict = {k:v for k,v in enumerate(new_cls)}
            except:
                sys.exit("--newclasses argument incorrectly formated. See README for more info")
        self.use_case = use_case

    def write_class_merger(self):
        print("Generating a class_merger.yaml file with data from %s and the new classes, [%s]\n"%(self.data_file, self.new_cls))
        # generate yaml file with new classes and old classes
        # for the purpose of merging classes 
        yaml_path = os.path.join(self.save,"class_merger.yaml")
        if os.path.isdir(yaml_path):
            ow = input("File %s already exists. Overwrite? Y/N")
            if ow.lower() == "y":
                print("Overwriting class_merger.yaml")
            else:
                sys.exit("ERROR: class_merger.yaml already exists and NOT overwriting. Select different path in --save argument.")

        # dictionary of {0: class0, 1: class1...}
        cls_dict = self.old_cls_dict
        #cls_dict = {k:v for k,v in enumerate(classes)}
        new_cls_dict = self.new_cls_dict

        data = {
            "new_classes": new_cls_dict,
            "old_classes": cls_dict
        }
        print("Saving %s"%yaml_path)

        with open(yaml_path, 'w') as outfile:
            yaml.dump(data, outfile, sort_keys=False)

    def read_class_merger(self):
        # reading yaml file with ,new_cls_idx on end of each line 
        # return dictionary of mappings to new classes

        new_class_mapping = {}

        try:
            with open(self.merge_file, 'r') as stream:
                data_loaded = yaml.safe_load(stream)

            to_merge = data_loaded['old_classes']
            new_classes = data_loaded['new_classes']
        except:
            sys.exit("%s failed to load. Check the file path and that \"old_classes\" and \"new_classes\" are present. If not, see README for correct class_merger.yaml format") 
        self.new_cls = []
        for cls in new_classes:
            self.new_cls.append(new_classes[cls])
        
        for cls in to_merge:
            if ',' not in str(to_merge[cls]):
                sys.exit("class_merger.yaml not in correct format. See README for further details")
            [n, merge] = str(to_merge[cls]).split(',')
            if int(merge) not in new_class_mapping:
                new_class_mapping[int(merge)] = [cls]
            else:
                new_class_mapping[int(merge)].append(cls)

        return new_class_mapping
    
    def change_classes(self, mapping):
        
        # location to save labels
        out_folder = os.path.join(self.save,"all_labels")

        if not os.path.isdir(out_folder):
            os.mkdir(out_folder)
        else:
            ow = input("WARNING: %s folder already exists. Overwrite contents? Y/N "%out_folder)
            if ow.lower() == "y":
                print("Overwriting..")
            else:
                sys.exit("ERROR: --save path already exists and overwrite NOT selected. Choose different --save path.")
        # add all labels to list
        labels_in = glob.glob(os.path.join(self.labels_in,"*.txt"))

        # for each label, open file, load each line, change first value based on mapping
        for label in labels_in:
            # grab just the file name
            save_name = label.split('/')[-1]
            
            new_lines = []
            f = open(label, "r")
            for line in f:
                # get first val
                first_space = line.find(" ")
                old_cls = int(line[0:first_space])
                new_cls = None
                for map_new in mapping:
                    if old_cls in mapping[map_new]:
                        new_cls = map_new
                    else:
                        continue
                
                if new_cls == None:
                    print(old_cls)
                    sys.exit("ERROR: Class merge error. Old class not found in %s"%label)
                new_lines.append("%s %s"%(str(new_cls),line[first_space+1:]))

            save_out = os.path.join(out_folder,save_name)
            with open(save_out, "w") as f:
                for line in new_lines:
                    f.write(line)


    def write_yaml(self):
        # generate yaml file with each class name
        yaml_path = os.path.join(self.save,"data.yaml")
        test_yaml_path = os.path.join(self.save,"test.yaml")
        # dictionary of {0: class0, 1: class1...}

        if os.path.isfile(yaml_path):
            ow = input("data.yaml already exists at %s. Overwrite? Y/N "%(yaml_path))
            if ow.lower() == "y":
                print("Overwriting data.yaml")
            else:
                sys.exit("ERROR: Not overwriting data.yaml. Please select different save path without exisiting yaml file.")
        
        if os.path.isfile(test_yaml_path):
            ow = input("test.yaml already exists at %s. Overwrite? Y/N "%(yaml_path))
            if ow.lower() == "y":
                print("Overwriting test.yaml")
            else:
                sys.exit("ERROR: Not overwriting test.yaml. Please select different save path without exisiting yaml file.")
   
        new_cls_enum = {k:v for k,v in enumerate(self.new_cls)}
        data = {
            "path":self.save,
            "train": "train",
            "val": "valid",
            "names": new_cls_enum
        }

        test = {
            "path":self.save,
            "train": "train",
            "val": "test",
            "names": new_cls_enum
        }


        with open(yaml_path, 'w') as outfile:
            yaml.dump(data, outfile, sort_keys=False)

        with open(test_yaml_path, 'w') as outfile:
            yaml.dump(test, outfile, sort_keys=False) 

    def run(self):

        # load new classes and data.yaml, write a fresh class_merger.yaml
        if self.use_case.lower() == "g":
            self.write_class_merger()
            sys.exit()

        # load class_merger.yaml to genereate mapping, open, edit, save all labels
        if self.use_case.lower() == "m":
            mapping = self.read_class_merger()
            self.change_classes(mapping)
            self.write_yaml()

def arg_parse():
    parser = argparse.ArgumentParser(
        description='Generate a class_merger.yaml or merge classes with a class_merger.yaml for labels that are in YOLO format')

    parser.add_argument("--newclasses", dest = "new_cls",
            help = "The class names to be merged to, separated by commas. Ex: cls0,cls1,cls2", 
            default = None, type = str, required=False)
    
    parser.add_argument("--data", dest = "data",
            help = "Path to data.yaml containing all original classes", 
            default = None, type = str, required=False)
    
    parser.add_argument("--labels_in", dest = "labels_in",
            help = "Path to labels to be merged", 
            default = None, type = str, required=False)
    
    parser.add_argument("--save", dest = "save",
            help = "Path to save location of merged labels - can be the same as --labels_in to overwrite", 
            default = None, type = str, required=True)
    
    parser.add_argument("--merge", dest = "merge_file",
            help = "Path to merge yaml file", 
            default = None, type = str, required=False)
    
    return parser.parse_args()


def main():
    args = arg_parse()
    use_case = input("Do you need to generate a merger file (G) or merge classes from existing class_merger.yaml (M)? G/M? ")
    # generating a merger file
    if use_case.lower() == "g":
        # check for new class list and data.yaml of classes
        if args.new_cls is None or args.data is None or args.save is None:
            sys.exit("Please ensure --newclasses, --data and --save arguments are included")
        if args.labels_in is not None or args.merge_file is not None:
            print("WARNING: --labels_in and --merge are being ignored\n") 

        labels_in = None
        save = args.save
        merge_file = None
        data = args.data
        new_cls = args.new_cls
    
    # merge from exisitng class_merger.yaml
    elif use_case.lower() == "m":
        if args.labels_in is None or args.save is None or args.merge_file is None:
            sys.exit("Please ensure --labels_in, --save and --merge arguments are included")
        if args.new_cls is not None or args.data is not None:
            print("WARNING: --newclasses and --data arguments are being ignored\n") 

        labels_in = args.labels_in
        save = args.save
        merge_file = args.merge_file
        data = None
        new_cls = None
    
    else:
        sys.exit("Please type G to generate a class_merger.yaml file or M to merge labels from an existing class_merger.yaml file")

    mc = mergeClasses(labels_in, save, merge_file, data, new_cls, use_case)
    mc.run()
    print("\nDONE")
if __name__=='__main__':
    main()
