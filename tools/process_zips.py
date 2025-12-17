'''
For processing lots of zip files downloaded from CVAT in COCO 1.0 format.
Takes all zip files in a folder, unzip and process the files into a useable format

Written by: Serena Mou
Date:       30/11/24

'''

import shutil
import glob
import os
import argparse

parser = argparse.ArgumentParser(description='Unzip all COCO 1.0 files in a folder, save contents into separate images/labels folders')
 
parser.add_argument("--root", dest = "root",
        help = "Path to folder where all zip files reside", default = None, type = str)

args = parser.parse_args()


root_path = args.root
zips_path = glob.glob(os.path.join(root_path,"*.zip"))
img_path = os.path.join(root_path,"images")
annot_path = os.path.join(root_path,"labels")
temp = os.path.join(root_path,"temp")

os.makedirs(temp, exist_ok=True)


for zip in zips_path:
    zip_name = zip.split('/')[-1]
    zip_name = zip_name.split('.')[0]
    # extract all to a temp folder
    shutil.unpack_archive(zip, temp)

    # move the annotaions/xxx.json file to RAW/labels/zip_name/annotations/xxx.json
    os.makedirs(os.path.join(annot_path,zip_name), exist_ok=True)
    shutil.move(os.path.join(temp,"annotations"),os.path.join(annot_path,zip_name))

    # move the images/default/xyz to RAW/images/xyz
    ims = glob.glob(os.path.join(temp,"images/default/**"))
    print(len(ims))
    for im in ims:
        shutil.move(im,img_path)
