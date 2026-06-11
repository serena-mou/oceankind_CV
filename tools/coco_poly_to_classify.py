#!/usr/bin/env python3

"""
Author: Serena Mou
Created: 09 July 2026

===
Converts the Polygons from CVAT in COCO format into classification labels with the images being of selected size. 
Majority of area inside image is within a polygon. 
If there are multiple jsons and the classes do not all match, a "all_classes_dict.yaml" will be required
===

"""
import cv2
import json
import os
import yaml
import glob
import csv
import sys
import argparse
import numpy as np
import math

class COCO2YOLOFBB():
    def __init__(self, json_file, save_location, im_path, classes_dict, imsz, overlap, poly_pct):

        # load in the json file
        #json_file = "/home/serena/Data/SCTLD/RAW/1_100/annotations/instances_default.json" 
        #f = open(json_file)

        #self.data = json.load(f)
        self.in_files = json_file #"/home/serena/Data/SCTLD/RAW/"
        self.save_location = save_location #"/home/serena/Data/SCTLD/Processed/"
        self.im_path = im_path
        self.all_classes_dict = {}
        if classes_dict is not None: 
            with open(classes_dict, 'r') as f:
                self.all_classes_dict = yaml.load(f, Loader=yaml.SafeLoader) 

        self.crop_size = imsz
        self.overlap = float(overlap/100)
        self.poly_pct = float(poly_pct/100)



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
            self.classes_dict = {k:v for k,v in enumerate(classes)}
            # print("all classes: ", len(classes), "\n")
            
            # List all the image filenames
            images = data["images"]
            self.img_names = [image["file_name"] for image in images]
            img_size = []
            for image in images:
                w = image["width"]
                h = image["height"]
                img_size.append([w,h])
            # For each annotation, get the class, image ID and the bbox
            annotations = data["annotations"]

            self.cls = [int(annotation["category_id"])-1 for annotation in annotations]
            self.img_ids = [int(annotation["image_id"])-1 for annotation in annotations]
            self.bbxs = [annotation["bbox"] for annotation in annotations]
            self.polys = [annotation["segmentation"] for annotation in annotations]
            self.im_sz = [img_size[img_id] for img_id in self.img_ids]
            # im_sz = [annotation["segmentation"]["size"] for annotation in annotations]
            # print(classes, img_names, cls, img_ids, bbxs, im_sz)
            return

            
        except:
            sys.exit("ERROR: json file in wrong format - check it is downloaded from CVAT in COCO 1.0 format, from polygon labels. ") 
    
  
    def write_label_summary(self, summary_dict):
        csv_path = os.path.join(self.save_location,"label_summary.csv")


        with open(csv_path, 'w') as outfile:
            writer = csv.writer(outfile)
            for key, value in summary_dict.items():
                writer.writerow([key,value])

    def poly_converter(self, poly):
        polygon = np.array(poly[0], np.int32)
        # print(polygon)
        polygon = polygon.reshape((int(len(polygon)/2)),2)
        #polygon = polygon.reshape((-1,1,2))
        
        return polygon

    def bbx_vertices(self, bbx):
        # Convert from CVAT bounding box [x top left, y top left, width, height]
        # to four corner vertices np.array([[top left], [top right], [bottom left], [bottom right]])

        xtl = int(bbx[0])
        ytl = int(bbx[1])
        w =   int(bbx[2])
        h =   int(bbx[3])

        # print("width: ", w)
        # print("height: ", h)
        top_left = np.array([xtl, ytl])
        top_right = np.array([xtl+w, ytl])
        bottom_left = np.array([xtl, ytl+h])
        bottom_right = np.array([xtl+w, ytl+h])
        vertices = np.vstack(([top_left],[top_right],[bottom_left],[bottom_right]))

        # vertices = vertices.reshape((-1,1,2))
        return vertices
    
    
    def bbx_converter(self, bbx_raw, im_sz):
        # given a bounding box in pixel format [x top left, y top left, width, height]
        # return in yolo format in normalised to image size [x middle, y middle, width, height]
        [xl, yl, w, h] = bbx_raw
        [fw, fh] = im_sz
        xn = (xl + (w/2))/fw
        yn = (yl + (h/2))/fh
        wn = (w/fw)
        hn = (h/fh)

        return [xn, yn, wn, hn]

    def crop_images(self):
        # make dirs for each class
        print(self.classes_dict)
        for val in self.classes_dict.values():
            
            path = os.path.join(self.save_location,val) 
            if not os.path.isdir(path):
                os.mkdir(path)
            else:
                ow = input("WARNING: Folder %s already exists. Overwrite or add to contents? [Y]/n "%(path))
                if ow.lower() == "y" or ow == "":
                    print("Continuing...")
                else:
                    sys.exit("ERROR: Not overwriting, use different path in --save argument")

        # for each image, loop through annotations, run window over bounding box
        for i,im in enumerate(self.img_names):
            get_name_str_end = im.rfind('.')
            im_basename = im[0:get_name_str_end]
            out_txt_name = im_basename + '.txt'
            all_im_idx = [j for j in range(len(self.img_ids)) if self.img_ids[j] == i]

            if len(all_im_idx) > 0:
                # print(all_im_idx)
                # handle different extensions, cvat issue?? eg. .jpg instead of .JPG
                path = glob.glob(os.path.join(self.im_path, im_basename+"*"))
                # print(path)
                if len(path)>1:
                    sys.exit("More than one image matched in image path.")
                
                image = cv2.imread(path[0])
                # for each annotation in this image 
                for idx in all_im_idx:
                    idx_class = self.cls[idx] 
                    if len(self.all_classes_dict.keys()) < 1:
                        full_list_class = idx_class 
                    else:
                        full_list_class = list(self.all_classes_dict.keys())[list(self.all_classes_dict.values()).index(classes[idx_class])]
                    
                    im_sz = self.im_sz[idx]
                    width = int(self.bbxs[idx][2])
                    height = int(self.bbxs[idx][3])
                    
                    vertices = self.bbx_vertices(self.bbxs[idx])
                    polygon = self.poly_converter(self.polys[idx])
                    poly_mask = np.zeros((im_sz[0],im_sz[1]))
                    # print(vertices)
                    # print(polygon)
                    cv2.fillPoly(poly_mask, [polygon], (1))

                    # start window of im size from top left
                    # number of windows in u direction
                    vs = math.floor((width - (self.crop_size*self.overlap))/(self.crop_size*(1-self.overlap)))                
                    # number of windows in v direction
                    us = math.floor((height - (self.crop_size*self.overlap))/(self.crop_size*(1-self.overlap)))                
                    # print(self.crop_size, self.overlap)
                    # print(us,vs)
                    # get the crop
                    for u in range(us):
                        for v in range(vs):
                            # print(u,v)
                            top_left = vertices[0]
                            row0 = int(top_left[1]+(u*(self.crop_size*(1-self.overlap))))
                            row1 = int(row0+self.crop_size)

                            col0 = int(top_left[0]+(v*(self.crop_size*(1-self.overlap))))
                            col1 = int(col0+self.crop_size)

                            # print(row0, row1, col0, col1)
                            # for each crop, check that mask percentage
                            cropped_mask = poly_mask[row0:row1, col0:col1] 
                            mask_pct = np.sum(cropped_mask)/(self.crop_size*self.crop_size)
                            if mask_pct > self.poly_pct:
                                
                                cropped_im = image[row0:row1, col0:col1]
                                image_name = '%s_%i_%i_%i.jpg'%(im_basename,idx,u,v)

                                cv2.imwrite(os.path.join(self.save_location,self.classes_dict[full_list_class],image_name), cropped_im)

                            '''
                            cv2.imshow("crop", cropped_im)
                            cv2.rectangle(image, vertices[0], vertices[3],(255,0,0),10)111
                            cv2.rectangle(image, (col0,row0), (col1,row1),(255,255,0),10)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()                            
                            
                            input()
                            '''
                    '''
                    cv2.imshow("Image", cv2.resize(poly_mask, (0,0), fx=0.2,fy=0.2))
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    #lines.append((full_list_class, xn, yn, wn, hn))
                    '''
        return
                   

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
            self.get_info(data)
            self.crop_images()

            
            ## Label summary
            #summary_dict = self.label_summary(classes, img_names, img_ids, cls, summary_dict)
        
            # self.write_label_summary(summary_dict) 
            
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

    parser.add_argument("--im_path", dest = "im_path",
            help = "Path to images", default = None, type = str, required=True)
    
    parser.add_argument("--classes", dest = "classes_dict",
            help = "Path to all_classes yaml file", 
            default = None, type = str, required=False)

    parser.add_argument("--imsz", dest = "imsz",
            help = "Size of cropped image for classification. Default is 640", 
            default = 640, type = int, required=False)

    parser.add_argument("--overlap", dest = "overlap",
            help = "Percentage of overlap between images. Default is 10", 
            default = 10, type = int, required=False)

    parser.add_argument("--poly_pct", dest = "poly_pct",
            help = "Percentage of cropped image that must be filled with labelled polygon. Default is 75", 
            default = 75, type = int, required=False)
     
    return parser.parse_args()


def main():
    args = arg_parse()

    #json_file = input("Path to JSON file or regex to files: ")
    #save_location = input("Path to save labels: ")
    test = COCO2YOLOFBB(args.json_file, args.save_location, args.im_path, args.classes_dict, args.imsz, args.overlap, args.poly_pct)
    test.run()
    print("DONE")

if __name__=='__main__':
    main()
