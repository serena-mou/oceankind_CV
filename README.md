# oceankind_CV
Computer vision pipeline for processing data from CVAT to train/validate/test with ultralytics for classification, detection and segmentation. Designed for Oceankind collaborators QUT, UVI, Berkeley and Point Blue. 

---
## 0. Get started

Clone this repository 

### Install miniconda
*If you have an existing conda installation, skip this step*

Follow instructions for your platform https://docs.anaconda.com/miniconda/

### Create conda environment and install ultralytics
As per https://docs.ultralytics.com/quickstart/#install-ultralytics 

```bash
conda create --name OK_CV 
conda activate OK_CV
conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 wandb scikit-learn
```

Instead of conda installing ultralytics, install from git for development

In an appropriate folder: 
```bash
# Clone the ultralytics repository
git clone https://github.com/ultralytics/ultralytics

# Navigate to the cloned directory
cd ultralytics

# Install the package in editable mode for development
pip install -e .
```

### Check installation
Make sure the install was successful. If CUDA is not available AND you have a CUDA enabled GPU, make sure it is set up correctly. 

Installing pytorch separately may help https://pytorch.org/get-started/locally/ 

```bash
python <path_to_this_repo>/tools/test_install.py
```
---
## 1. Data Preparation
### CVAT Segment Anything Mask annotation to YOLO Bounding box format:
For converting CVAT Segment Anything mask annotations to YOLO bounding box format for training:

1. Go to Project -> Open Task -> Export annotations in COCO 1.0 format

    *CVAT has YOLOv8 bounding boxes built in as a format, however it does not convert from segmentation labels to bounding boxes. Skip this conversion step this if the annotations are already in bounding box format*

2. Download the zip and extract. It should be in the format annotation/instances_default.json

    If you have many different CVAT tasks for the same model, extract all into the same folder in following format:
```
└── Raw_labels
    ├── 0-100
        └── annotations
            └── instances_default.json
    ├── 101-200
        └── annotations
            └── instances_default.json
    └── 201-300
        └── annotations
            └── instances_default.json          
```

3. To convert to YOLO format:

**Example Use**

```bash
# Run conversion script
python tools/coco_to_yolo_format.py --json <path to Dataset>/Raw_labels/*/*/*.json --save <path to Dataset>/<Dataset name>/
```

**Options**
- ```--json str``` Path to a json file OR a regex to all the json files
- ```--save str``` Path to a folder to save the labels in

**Outputs**
- ```<save>/all_labels/``` Folder in --save with bounding box labels for each image in YOLO format
- ```<save>/data.yaml``` File with the data for YOLO training
- ```<save>/test.yaml``` File with the data for YOLO testing

### CVAT Segment Anything Mask annotation to YOLO segmentation format:
For training segmentation models, download from CVAT in YOLOv8 Segmentation 1.0 format. Once unzipped, the file structure is as follows:
```
└── Dataset
    ├── labels
        └── train
            ├── im1.txt 
            ├── im2.txt
            └── imn.txt
    ├── data.yaml
    └── train.txt        
```
To standardise file structure for merging, data splitting and training:
1. Move all label files into a folder named "all_labels" (no nested folders)
2. Edit data.yaml to point the "path" to your Dataset, edit "train" and add "val". See examples/data.yaml
3. Copy the data.yaml file into a file named "test.yaml". Change "val" to point to "test". See examples/test.yaml

The data should be structured as follows:
```
└── Dataset
    ├── all_labels
        ├── im1.txt 
        ├── im2.txt
        └── imn.txt
    ├── data.yaml
    └── test.yaml      
```


## 1.5. (Optional) Merge Classes

If classes need to be merged, use this script. Skip if original classes are suitable for your needs. There are two functionalities.

- Generate class_merger.yaml. If one is not existing, the code will generate one to be edited. 
- Merge classes based on a class_merger.yaml file. 

**Example Use**

```bash
# Run merging script to generate class_merger.yaml
python tools/merge_classes.py --save <path to Dataset>/merged_Animals/ --data <path to Dataset>/data.yaml --newclasses alive,not_alive

# The script will prompt:
Do you need to generate a merger file (G) or merge classes from existing class_merger.yaml (M)? G
# Type "G" and Enter
```

See below for how to correctly edit class_merger.yaml

```bash
# Run merging script to merge classes
python tools/merge_classes.py --save <path to Dataset>/merged_Animals/ --merge <path to Dataset>/merged_Animals/class_merger.yaml --labels_in <path to Dataset>/all_labels 

# The script will prompt:
Do you need to generate a merger file (G) or merge classes from existing class_merger.yaml (M)? M
# Type "M" and Enter
```


**Options**
- ```--save str``` Path to a folder to save to. If it is the same as labels_in, the label files will be overwritten. 
- ```--data str``` Path to data.yaml file from CVAT YOLO download or generated from coco_to_yolo_format.py. This contains all the "old_classes".
- ```--newclasses str``` The class names to be merged to, separated by commas. Ex: cls0,cls1,cls2.
- ```--merge str``` Path to merge yaml file
- ```--labels_in str``` Path to label files to be merged

**Outputs**
    
For "generate a merger file (G)" option: 
- ```<save>/class_merger.yaml/``` File to help merging of classes. See below on how to correctly edit this file.

For "merge classes from existing class_merger.yaml (M)" option:

- ```<save>/all_labels/``` Folder in --save with all label files where the class has been updated according to class_merger.yaml.
- ```<save>/data.yaml``` File with the data information for training with updated new classes.
- ```<save>/data.yaml``` File with the data information for testing with updated new classes.


### Editing a class_merger.yaml
To associate an "old_class" to a "new_class", add the number associated to the new class at the end of the line. For example, the examples/class_merger.yaml should be edited to:

<table>
<tr>
<th> Before </th>
<th> After </th>
</tr>
<tr>
<td>

```yaml
new_classes:
  0: alive
  1: not_alive
old_classes:
  0: dog
  1: cat
  2: person
  3: laptop
```
</td>
<td>

```yaml
new_classes:
  0: alive
  1: not_alive
old_classes:
  0: dog,0
  1: cat,0
  2: person,0
  3: laptop,1
```

</td>
</tr>
</table>

Save the yaml file and rerun tools/coco_to_yolo_format.py, this time providing the path to the edited class_merger.yaml

---
## 2. Split data

To process the data into train/validate/test splits for YOLO, data needs to be in the following format:
```
└── Dataset
    ├── all_labels
        ├── im1.txt
        └── im2.txt      
    ├── all_images
        ├── im1.png
        └── im2.png           
    └── data.yaml   
```

*If the labels were generated by the coco_to_yolo_format.py script, they will already be in an `all_labels` folder.*

*If the annotations were in bounding box format, download annotations from CVAT in YOLOv8 Detection 1.0 format and extract into a folder named `all_labels`.*

*Images need to placed into `all_images` folder.*


### Create training data splits
For model training the combined data sets need to be split into training, validation and testing - YOLO tools expect the following structure:
```
└──/Dataset/
    
    ├──/train
	    ├──/images
	    └──/labels
    ├──/valid
	    ├──/images
	    └──/labels
    └──/test/
	    ├──/images
	    └──/labels
```
The `bal_train_test_split.py` tool faciliates this, with a set random seed to repeatably generate splits given the same inputs. The tool attempts to preserve class ratios present in inputs, see `train_test_split.py` for original version that did not implement this.

**Example usage:**
```
python tools/bal_train_test_split.py --src /Dataset --valid 0.2 --test 0.1
```

**Options:**
- `--src str` Source folder to search, expects `/src/all_images`, `/src/all_labels`
- `--valid float` Percentage of dataset to use for validation `default=0.2`
- `--test float` Percentage of dataset to use for testing `default=None`
- `--dump int` Optionally remove n unlabelled images from dataset `default=None`


**Outputs:**
- `/src/train/images/` Randomly selected training images
- `/src/train/labels/` YOLO format labels corresponding to train
- `/src/valid/images/` Randomly selected training images
- `/src/valid/labels/` YOLO format labels corresponding to valid
- `/src/test/images/` Randomly selected training images
- `/src/test/labels/` YOLO format labels corresponding to test

---
## 3. Training

### (Optional) Visualisation
Create a wandb account (www.wandb.ai) and login in terminal for visualisations
```bash
wandb login
```

### Train
Edit the `training/train.py` to the parameters you need, then run **from the folder for results**. The default model is `yolov8m.pt`. See https://docs.ultralytics.com/models/yolov8/#performance-metrics for other Yolov8 pre-trained models. `data.yaml` generated by the `coco_to_yolo_format.py` is correctly formatted for the split and training. One may need to be generated/edited if downloaded bounding boxes straight from CVAT. See `examples/data.yaml` for an example file. 

**Example usage:**
```bash
python <path_to_this_repo>/training/train.py --src /Dataset/data.yaml --name Animal_Train
```
***The outputs will be saved to the pwd (present working directory- path the script is being run from).***


**Options:**
- `--src str` Source YOLO yaml file describing train dataset
- `--name str` Model name for saving

**Outputs:**
- `/pwd/OK_CV/name/` Model training stats and outputs


## 4. Testing

Test the weights on unseen test data. A `test.yaml` will have been generated by the `coco_to_yolo_format.py`. If one needs to be generated, see `examples/test.yaml` to use as template. It is important to use the `test.yaml`, not the previously used `data.yaml`

**Example usage:**
```bash
python <path_to_this_repo>/training/test.py --src /Dataset/test.yaml --weights /Dataset/OK_CV/Animal_Train/weights/best.pt --name Animal_Test
```
***The outputs will be saved to the pwd (present working directory- path the script is being run from).***


**Options:**
- `--src str` Source YOLO yaml file describing test dataset
- `--weights str` Path to the model weight file for evaluation
- `--name str` Model name for saving

**Outputs:**
- `/pwd/OK_CV/name/` Model test stats and outputs


---
## Free labelling software

CVAT is free, to a point. For completely free labelling software, we can use label-studio (previously labelImg). label-studio does not currently support SAM so would be more suited to bounding box level annotations. It can produce segmentation, however you will need to hand paint the mask or hand draw the polygon. This method has been tested for Bounding Boxes.  

1. Create a new conda instance - label-studio has different version dependencies than ultralytics. 

```bash
# If you're still in the OK_CV environment
conda deactivate 

# Install
conda create --name label-studio
conda activate label-studio
conda install psycopg2
pip install label-studio

# Spin up
label-studio
```

2. Follow label-studio instructions to set up project, upload data, set up labelling interface and label names, and label your data. 

3. Once fully labelled, from the Project page, select Export and select YOLO format. 

4. Unzip the data. 

5. Run `label_studio_convert.py`

**Example usage:**
```bash
python tools/label_studio_convert.py --src /Downloads/<unzipped label studio folder> --dest /Data_folder/Animals
```

**Options:**
- `--src str` Path to unzipped label-studio folder
- `--dest str` Path to desired dataset location

**Outputs:**
- `<dest>/all_images` Folder with all the images for the dataset 
- `<dest>/all_labels` Folder with all the label files for the dataset 
- `<dest>/data.yaml` YAML file for YOLO training script 
- `<dest>/test.yaml` YAML file for YOLO test script 


6. Follow instructions starting from `2. Split data`


---
# Helpers:

To visualise bounding box annotations, use the visualise_bb_annots.py. Only the --src argument is required if the data is organised as above. If you would like to view just a smaller subset of image/label pairs, paste the desired label files into a separate folder and add the folder name to --label_folder. The img_folder can be left as default, just the images matching the label files in the smaller subset will be shown. 

**Example usage**
```bash
python tools/visualise_bb_annots.py --src <path to dataset root dir>
```

**Options**
- `--src str` Path to the root directory of your data
- `--img_folder str` (Optional) If the folder of images is not the default <src>/all_images, add name of folder
- `--label_folder str` (Optional) If the folder of labels is not the default <src>/all_labels, add name of folder
- `--data str` (Optional) If the yaml file with class names is not the default <src>/data.yaml, add name of file
- `--scale float` (Optional) Scale the image size up or down. Default is 0.5.

**Outputs:**
OpenCV window showing images with bounding boxes and name of classes. Press any key to go to the next image, press ESC to escape. 

