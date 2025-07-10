#!/usr/bin/env python3

"""
Author: Serena Mou
Created: 8th April, 2025 

===
Steal segmentations and place into different "background" images. **NOT CHECKED FOR >1 SEG PER IM**
===

"""
import glob
import cv2
import numpy as np
import os
import shutil

class stealSeg():
    def __init__(self):

        # Segmentations
        self.labels_in = "/home/serena/Data/Fish/Raw/fake_fish/seg_label/labels/train/*"
        self.images_in = "/home/serena/Data/Fish/Raw/fake_fish/seg_label/images/train/*"


        # Empty images to paste segmentations into
        self.background_ims = "/home/serena/Data/Fish/Raw/fake_fish/zed_blank/zed/*"

        # Save new ims and labels
        self.save_out = "/home/serena/Data/Fish/Processed/April_25_fake/"

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

    def get_name_from_path(self, path:str, end:bool):
        name = path.split('/')[-1]
        if end:
            return name
        base = name[0:name.rfind('.')]
        return base
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
                        raw_seg = line[first_space+1:]
                        seg_poly = self.seg_converter(raw_seg, image.shape)
                        x = np.insert(seg_poly, 0, cls, axis=0)
                        seg_dict[im] = x
            
        return seg_dict
    

    def copy_seg(self):
        # for each background, copy in segmentation/s

        all_bg = glob.glob(self.background_ims)
        seg_dict = self.get_seg_dict()
        for bg in all_bg:
            bg_im = cv2.imread(bg)
            # randomly select a segmentation
            ri = np.random.randint(len(seg_dict))
            seg = list(seg_dict.keys())
            im_path = seg[ri]
            ## ** Add rand for > 1 per im **
            label = seg_dict[im_path]
            
            # print(im_path, label)

            im_in = cv2.imread(im_path)
            zeros = np.zeros(im_in.shape[:-1]).astype(im_in.dtype)
            m0 = cv2.fillPoly(zeros, np.int32([label[1:]]), 255)
            #fish_mask = cv2.bitwise_and(im_in, m0)

            sel = zeros!=255
            im_in[sel] = bg_im[sel]

            # generate im name 
            bg_name = self.get_name_from_path(bg,True) 
            seg_name = self.get_name_from_path(im_path,False)
            new_name = seg_name+bg_name

            # save image
            cv2.imwrite(os.path.join(self.save_out,"images",new_name),im_in)
            
            # copy label file 
            label_path = self.labels_in[0:-2]
            src = os.path.join(label_path,seg_name+".txt")
            dest = os.path.join(self.save_out,"labels",new_name[0:-4]+'.txt')
            shutil.copy(src, dest)
            # cv2.imshow("mask",im_in)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        # input()


def main():
    theft = stealSeg()
    theft.copy_seg()

if __name__=='__main__':
    main()




