# split image classification folders into train/val/test sets
# Written by: Serena Mou
# Date:       4th Aug 2026
import glob
import shutil
import os
import random
import argparse


def arg_parse():

    parser = argparse.ArgumentParser(
        description="sort images into folders according to keypress"
    )

    parser.add_argument("-src","--source", dest="source",
        help = "Path to folder of images to sort", default=None, type=str, required=True
    )
    
    parser.add_argument("-s","--split", dest="split",
        help = "Fraction to split train/valid/test set. Must sum to 1.0. Default 0.7,0.2,0.1", default="0.7,0.2,0.1", type=str, required=False
    )

    return parser.parse_args()


args = arg_parse()

root = args.source
# save location

classes = os.listdir(root)

out_data = os.path.join(root, "split")
if os.path.isdir(out_data): 
    print("Directory %s already exists. Continue? Y/N"%out_data)
    if input().lower() != 'y':
        sys.exit("Exiting")
os.makedirs(out_data, exist_ok=True)


# split locations
split = [float(x) for x in args.split.split(',')]

if sum(split) != 1.0:
    sys.exit("Split fraction does not sum to 1. Exiting. ")

split_name = ['train', 'valid', 'test']

random.seed(5)

for spt_f in split_name:
    if not os.path.isdir(os.path.join(out_data,spt_f)):
        os.mkdir(os.path.join(out_data,spt_f))
    for cl in classes:
        pth =  os.path.join(out_data,str(spt_f),str(cl))
        if not os.path.isdir(pth):
            os.mkdir(pth)


# for each folder, split into percentage and move
for cl in classes:
    patches = glob.glob(os.path.join(root,cl,'*'))

    # shuffle that bitch
    random.shuffle(patches)
    all_len = len(patches)
    idx = [int(split[0]*all_len), int(split[0]*all_len)+int(split[1]*all_len)]
    Train = patches[0:idx[0]]
    Val = patches[idx[0]:idx[1]]
    Test = patches[idx[1]:]
    # print(cl, len(Train), len,(Val), len(Test)) 

    for im in Train:
        im_name = im.split('/')[-1]
        out_path = os.path.join(out_data, 'train', cl,im_name)
        shutil.copy(im,out_path)

    for im in Val:
        im_name = im.split('/')[-1]
        out_path = os.path.join(out_data, 'valid', cl,im_name)
        shutil.copy(im,out_path)

    for im in Test:
        im_name = im.split('/')[-1]
        out_path = os.path.join(out_data, 'test', cl,im_name)
        shutil.copy(im,out_path)