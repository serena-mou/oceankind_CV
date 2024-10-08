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
For converting CVAT Segment Anything mask annotations to YOLO format for training:

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
python tools/coco_to_yolo_format.py --json /Data/CVAT_example/Raw_labels/*/*/*.json --save /Data/Processed/CVAT_example/ --merge /Data/CVAT_example/class_merger.yaml --newclasses living,not_living
```

**Options**
- ```--json str``` Path to a json file OR a regex to all the json files
- ```--save str``` Path to a folder to save the labels in
- ```--merge str``` (Optional) Path to the class_merger.yaml if merging from existing yaml file. Only used if merging and using existing merge file. 
- ```--newclasses str``` (Optional) A list of classes to be merged into, separated by commas. Only used if merging and new merge file needs to be generated. 

The code will prompt Y/N to determine use case. Type Y or N and press enter. 
```bash
Do you need to merge classes? Y/N: 
## if No, the script will complete, --merge and --newclasses are ignored 

## if Yes:

Do you have an existing merge file? Y/N:
## if No, --newclasses is required, --merge is ignored
## The class merger yaml will be generated and script will exit.
## You will need to edit the yaml file and save. Read class_merger section for more information.
## Run conversion script again, this time entering Y for existing merge file. 

## if Yes:
## --merge is required, --newclasses is ignored
```

**Outputs**
- ```<save>/all_labels/``` Folder in --save with bounding box labels for each image in YOLO format
- ```<save>/data.yaml``` File with the data information for training
- ```<save>/class_merger.yaml``` (Optional) File with class merging information will be generated if requested

### class_merger
---
The class_merger.yaml is automatically generated by the tools/coco_to_yolo_format.py. See an example in examples/class_merger.yaml

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
python <path_to_this_repo>/training/train.py --src /Dataset/data.yaml --name Animal_Model
```
***The outputs will be saved to the pwd (present working directory- path the script is being run from).***


**Options:**
- `--src str` Source YOLO yaml file describing train dataset
- `--name str` Model name for saving

**Outputs:**
- `/pwd/OK_CV/name/` Model training stats and outputs


