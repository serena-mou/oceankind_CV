# Process the zip files into a useable format

import shutil
import glob
import os

root_path = "/home/serena/Data/SCTLD/Jan25/RAW/"
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
