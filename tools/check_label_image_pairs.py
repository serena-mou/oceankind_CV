'''
Take in a label folder and image folder, make new image folder and copy only images that match labels
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
    parser = argparse.ArgumentParser(description='Check folder of labels, paste matching images into different folder')

    parser.add_argument("--img_folder", dest = "img_folder",
            help = "Path to folder of images", default = "all_images", type = str)
    parser.add_argument("--label_folder", dest = "label_folder",
            help = "Path of folder of labels", default = "all_labels", type = str)
    parser.add_argument("--save_imgs", dest = "save_imgs",
            help = "Path to save images", default = "images_out", type = str)


    return parser.parse_args()



def main():
    args = arg_parse()
    os.makedirs(args.save_imgs, exist_ok=True)

    all_labels = glob.glob(os.path.join(args.label_folder,'*'))

    for i in range(0,len(all_labels)):
        label = all_labels[i]
        # get matching image

        # img name
        label_name = label.split('/')[-1]
        img_name = label_name[:-4]#label_name.split('.')[0]
        print(img_name)
        img_path = glob.glob(os.path.join(args.img_folder,img_name+"*"))
        if len(img_path) > 1:
            print("WARNING: label name matched to more than one img")
        if len(img_path) == 0:
            print("WARNING: image not found for %s"%label)
        else:
            im_name_ext = img_path[0]
            im_name_ext = im_name_ext.split('/')[-1]
            #print(img_path)
            #print(os.path.join(args.save_imgs,im_name_ext))
        shutil.copy(img_path[0],os.path.join(args.save_imgs,im_name_ext))


if __name__=='__main__':
    main()
