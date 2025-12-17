#!/usr/bin/env python3

"""
Author: Serena Mou
Created: 8th April, 2025 

===
Steal segmentations and place into different "background" images. 
Usage example:
python3 ~/repos/oceankind_CV/tools/steal_segmentations.py --labels_in to_steal/labels/ 
--images_in to_steal/images/ --theft_list 1,2,3,4 --bg_images blank_zed_bg/ --save thefted
===

"""
import glob
import cv2
import numpy as np
import os
import shutil
import argparse


class stealSeg():
    def __init__(self, labels_in, images_in, theft_list, bg_images, save):

        # Segmentations
        self.labels_in = os.path.join(labels_in,"*") # "/home/serena/Data/Fish/Processed/SEPT25/all_ims_all_labels_steal_EPST/train/labels/*"
        self.images_in = os.path.join(images_in,"*") #"/home/serena/Data/Fish/Processed/SEPT25/all_ims_all_labels_steal_EPST/train/images/*"

        # List of classes to steal
        self.theft_list = [int(i) for i in theft_list.split(',')]#[1,2,3,4]

        # Empty images to paste segmentations into
        self.background_ims = os.path.join(bg_images, "*")#"/home/serena/Data/Fish/Processed/SEPT25/steal_fish_data/blank_zed_bg/*"

        # Save new ims and labels
        save_out = save# "/home/serena/Data/Fish/Processed/SEPT25/all_ims_all_labels_steal_EPST/stolen_train_weighted/"
        self.save_ims = os.path.join(save_out,"images")
        self.save_labs = os.path.join(save_out,"labels")
        os.makedirs(self.save_ims, exist_ok=True)        
        os.makedirs(self.save_labs, exist_ok=True)        

        np.random.seed(3)

    def seg_converter(self,seg, im_sz):
        ## Convert yolo segmentation format to (x,y) pairs of polygon for segmentation
        seg = seg.split(' ')
        # print(im_sz)
        scale = np.array([im_sz[1], im_sz[0]])
        # print(scale)
        data = [float(s) for s in seg]
        l = int(len(data)/2)
        data = np.reshape(data,(l,2))*scale
        # print(data.astype(int))
        data = data.astype(int)
        out = []
        for d in data:
            out.append((d[0],d[1]))
        return np.asarray(list(out))
    
    def poly_converter(self, label, rx, ry, bg_shape):
        cls = label[0][0]
        poly = label[1:]
        rect = cv2.boundingRect(poly)
        x,y,w,h = rect
        #x_diff = x-rx
        #y_diff = y-ry
        poly_bg = poly-[x, y]
        poly_bg = poly_bg+[rx, ry]
        # print(cls, x, y, rx, ry, bg_shape)
        # print(poly)
        # print(poly_bg)
        scale = np.array([bg_shape[1], bg_shape[0]])
        label_bg = poly_bg/scale
        if np.min(label_bg) < 0 or np.max(label_bg) > 1:
            print(label_bg)
            input()
        # print(label_bg)
        # input()
        data = np.reshape(label_bg, (2*len(label_bg),1))
        line = str(cls)
        for d in data:
            line = line+' '+"%.6f"%np.clip(d[0],0,1)
        return line


    def get_name_from_path(self, path:str, end:bool):
        name = path.split('/')[-1]
        if end:
            return name
        base = name[0:name.rfind('.')]
        return base
    
    # def get_path_from_name(self, root, name):


    
    def get_seg_dict(self):

        # return a dictionary where key:val is image_path:[label0,label1...]
        seg_dict = {}
        all_ims = glob.glob(self.images_in)
        all_labels = glob.glob(self.labels_in)
        
        for im in all_ims:
            image = cv2.imread(im)
            # get labels for this im
            base = self.get_name_from_path(im, False)
            label_name = base + '.txt'
            for label in all_labels:
                
                if label.split('/')[-1] == label_name:
                    # open label file
                    f = open(label,"r")
                    for line in f:
                        line.rstrip()
                        first_space = line.find(" ")
                        cls = int(line[0:first_space])
                        if cls in self.theft_list:
                            raw_seg = line[first_space+1:]
                            seg_poly = self.seg_converter(raw_seg, image.shape)
                            x = np.insert(seg_poly, 0, cls, axis=0)
                            if im not in seg_dict:
                                seg_dict[im] = [x]
                            else:
                                seg_dict[im].append(x)
                            
            
        return seg_dict
    

    def copy_seg(self):
        # for each background, copy in segmentation/s

        all_bg = glob.glob(self.background_ims)
        seg_dict = self.get_seg_dict()
        # print(seg_dict)
        for i in range(0,5):
            for bg in all_bg:

                # background image
                bg_im = cv2.imread(bg)
                
                # randomly select a segmentation
                ri = np.random.randint(len(seg_dict))
                seg = list(seg_dict.keys())
                ## ** Add rand for > 1 per im **
                im_path = seg[ri]
                rand_lab = np.random.randint(len(seg_dict[im_path]))
                label = seg_dict[im_path][rand_lab]
                # segmentation image
                seg_im = cv2.imread(im_path)
                poly = label[1:]
                rect = cv2.boundingRect(poly)
                x,y,w,h = rect
                
                seg_mask = np.zeros(seg_im.shape[:2],np.uint8)
                _ = cv2.fillPoly(seg_mask, [poly], 255)
                fish_mask = seg_mask[y:y+h, x:x+w].copy()

                # select a region of interest in background
                # check that the fish can fit into the background
                (bg_y_max, bg_x_max, d) = bg_im.shape
                (fish_y, fish_x) = fish_mask.shape
                if bg_y_max > fish_y and bg_x_max > fish_x:
                    # randomly select roi
                    x_diff = bg_x_max - fish_x
                    y_diff = bg_y_max - fish_y
                    rand_x = np.random.randint(x_diff)
                    rand_y = np.random.randint(y_diff)
                    #print(rand_x, rand_y)
                    roi = bg_im[rand_y:h+rand_y, rand_x:w+rand_x]
                    #print(bg_im.shape, fish_mask.shape, rand_x, rand_y, x, y, w, h)
                    #print(rand_y,h+rand_y, rand_x,w+rand_x)
                    seg_im_crop = seg_im[y:y+h, x:x+w].copy()

                    ## maybe some hue shifting?
                    # get average hue of bg
                    # cvt to hsv
                    '''
                    roi_bg_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    bg_hue_mean = cv2.mean(roi_bg_hsv)

                    roi_fg_hsv = cv2.cvtColor(seg_im_crop, cv2.COLOR_BGR2HSV) 
                    fg_hue_mean = cv2.mean(roi_fg_hsv)
                    conv_fg_mean = np.asarray(fg_hue_mean)*[1.972, 0.392, 0.392, 1]
                    conv_bg_mean = np.asarray(bg_hue_mean)*[1.972, 0.392, 0.392, 1]
                    hue_diff = int((conv_fg_mean[0]-conv_bg_mean[0])/4)

                    hue,sat,val = cv2.split(roi_fg_hsv)
                    h_new = np.mod(hue-hue_diff, 180).astype(np.uint8)
                    fg_hsv_new = cv2.merge([h_new,sat,val])
                    fg_shifted_bgr = cv2.cvtColor(fg_hsv_new, cv2.COLOR_HSV2BGR)

                    # cv2.imshow("ahh", fg_shifted_bgr)
                    # cv2.waitKey(0)
                    
                    # input()
                    '''
                    inv_fish_mask = cv2.bitwise_not(fish_mask)
                    roi_bg = cv2.bitwise_and(roi, roi, mask=inv_fish_mask)
                    # roi_fg = cv2.bitwise_and(seg_im_crop, seg_im_crop, mask=fish_mask)
                    # roi_fg = cv2.bitwise_and(fg_shifted_bgr, fg_shifted_bgr, mask=fish_mask)
    
                    
                    # roi_final = cv2.add(roi_bg,roi_fg)
                    roi_fg_weighted = cv2.addWeighted(seg_im_crop, 0.4, roi, 0.6, 0)
                    roi_fg = cv2.bitwise_and(roi_fg_weighted, roi_fg_weighted, mask=fish_mask)

                    roi_final = cv2.add(roi_bg,roi_fg)
                    

                    combined_final = bg_im.copy()
                    combined_final[rand_y:h+rand_y, rand_x:w+rand_x] = roi_final 
                    
                    # print("BG: ", bg_im.shape)
                    # print("fish: ", fish_mask.shape)

                    
                    # cv2.imshow("ah",combined_final)
                    # cv2.waitKey(0)
                    # input()

                    # generate im name 
                    bg_name = self.get_name_from_path(bg,True) 
                    seg_name = self.get_name_from_path(im_path,False)
                    new_name = seg_name+str(rand_lab)+bg_name


                    # save image
                    cv2.imwrite(os.path.join(self.save_ims,new_name),combined_final)
                    

                    ## update label
                    line = self.poly_converter(label,rand_x, rand_y, bg_im.shape)
                    dest = os.path.join(self.save_labs,new_name[0:-4]+'.txt')
                    with open(dest,'w') as f:
                        f.write("%s\n"%line)

                    # copy label file 
                    # label_path = self.labels_in[0:-2]
                    # src = os.path.join(label_path,seg_name+".txt")
                    # dest = os.path.join(self.save_labs,new_name[0:-4]+'.txt')
                    # shutil.copy(src, dest)
                    # cv2.imshow("mask",im_in)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
            else:
                continue
            # input()

def arg_parse():
    parser = argparse.ArgumentParser(
        description='Steal segmentations and paste them into "blank" backgrounds')
    
    parser.add_argument("--labels_in", dest = "labels_in",
            help = "Path to labels to be stolen", 
            default = None, type = str, required=True)
 
    parser.add_argument("--images_in", dest = "images_in",
            help = "Path to images to be stolen", 
            default = None, type = str, required=True)
    
    parser.add_argument("--theft_list", dest = "theft_list",
            help = "List of all the classes to steal from. Separate class id with a comma. Eg. 1,4,5", 
            default = None, type = str, required=True)
    
    parser.add_argument("--bg_images", dest = "bg_images",
            help = "Path to background images", 
            default = None, type = str, required=True)
 
    parser.add_argument("--save", dest = "save",
            help = "Path to save location of new images and labels", 
            default = None, type = str, required=True)
    
    return parser.parse_args()

def main():
    args = arg_parse()
    theft = stealSeg(args.labels_in, args.images_in, args.theft_list, args.bg_images, args.save)
    theft.copy_seg()

if __name__=='__main__':
    main()




