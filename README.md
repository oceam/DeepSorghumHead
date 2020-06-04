# README: DeepSorghumHead

This is a support page of paper published at "Plant Phenomics" as an Original Research. 

> Ghosal,S., Zheng, B., Chapman, S.C., Potgieter, A.B., Jordan,D.R., Wang,X., Singh, A.K., Singh, A., Hirafuji, M., Ninomiya, S., Ganapathysubramanian, B., Sarkar, S., Guo, W.(2019). A weakly supervised deep learning framework for sorghum head detection and counting. Plant Phenomics.https://spj.sciencemag.org/plantphenomics/2019/1525874/

Please cite this paper below if you used the dataset provided here.

We provide three different types of in Supplementary Materials: 

1. Supplementary Materials 1:
   1)	Original image dataset corresponded Dot labeled data (Bounding box). 

2. Supplementary Materials 2:
   1) Source code of proposed method
   2) Finetuned model used in this paper

To download the dataset, please fill out a simple Form below:

https://docs.google.com/forms/d/e/1FAIpQLSdHQ0aXJUvjba4X5qnqynwFbK3YDlGOwEbUZjkfVPtfSJHt7w/viewform

The download link will be sent to your email once the form is completed. 

# Bibtex Info:
```
@article{ghosal2019weakly,
  
  title={A Weakly Supervised Deep Learning Framework for Sorghum Head Detection and Counting},
  
  author={Ghosal, Sambuddha and Zheng, Bangyou and Chapman, Scott C and Potgieter, Andries B and Jordan, David R and Wang, Xuemin and Singh, Asheesh K and Singh, Arti and Hirafuji, Masayuki and Ninomiya, Seishi and others},

  journal={Plant Phenomics},

  volume={2019},
  
  pages={1525874},
  
  year={2019},
  
  publisher={AAAS}

}
```
# TRAINED MODEL located in DeepSorghumHead_Keras/bin/inf_models/

# Running the ToolChain: (navigate to folder where your .py file is located first, open a terminal window and execute following syntax)

1. For Training (from scratch or using pre-trained model):

CUDA_VISIBLE_DEVICES=<### ENTER GPU DEVICE ID ###> python train_custom.py --backbone resnet50 --weights <### ENTER PATH TO PRE_TRAINED MODEL WEIGHTS ###> --batch-size 10 --epochs 200 --steps <### NUMBER OF STEPS - THIS IS DETERMINED BY YOUR TRAINING DATA SIZE AND BATCH SIZE, STEPS = TRAINING DATA SIZE/BATCH SIZE ###> --snapshot-path <### ENTER PATH TO SAVE NEW MODEL WEIGHTS ###> --tensorboard-dir <### ENTER TENSORBOARD LOG PATH ###> --random-transform --image-min-side 1600 --image-max-side 2400 csv <### ENTER PATH TO TRAINING DATA (.csv FILE) ###> <### ENTER PATH TO ANNOTATION MAP (THE 'annotation_map.csv' FILE) ###>

2. For converting your Trained model to an Inference Model (this must be done before you can test your model on test data):

python convert_model.py <### ENTER PATH TO PRE_TRAINED MODEL WEIGHTS ###> <### ENTER PATH TO SAVE CONVERTED MODEL WEIGHTS ###>

3. Evaluate the Trained Model on Test Data:

python evaluate.py --backbone resnet50 --max-detections 600 --save-path <### ENTER PATH FOR OUTPUT DIRECTORY ###> csv <### ENTER PATH WHERE TEST DATA IS LOCATED (.csv FILE) ###> <### ENTER PATH TO ANNOTATION MAP (THE 'annotation_map.csv' FILE) ###> <### ENTER PATH WHERE TRAINED INFERENCE MODEL IS LOCATED ###>

# Base Code originally from https://github.com/fizyr/keras-retinanet Following is repeated from the README file of the Base Code repository:

# Generating your own custom CSV datasets from images:
"The CSVGenerator provides an easy way to define your own datasets. It uses two CSV files: one file containing annotations and one file containing a class name to ID mapping.

Annotations format
The CSV file with annotations should contain one annotation per line. Images with multiple bounding boxes should use one row per bounding box. Note that indexing for pixel values starts at 0. The expected format of each line is:

path/to/image.jpg,x1,y1,x2,y2,class_name
Some images may not contain any labeled objects. To add these images to the dataset as negative examples, add an annotation where x1, y1, x2, y2 and class_name are all empty:

path/to/image.jpg,,,,,
A full example:

/data/imgs/img_001.jpg,837,346,981,456,cow
/data/imgs/img_002.jpg,215,312,279,391,cat
/data/imgs/img_002.jpg,22,5,89,84,bird
/data/imgs/img_003.jpg,,,,,
This defines a dataset with 3 images. img_001.jpg contains a cow. img_002.jpg contains a cat and a bird. img_003.jpg contains no interesting objects/animals.

Class mapping format
The class name to ID mapping file should contain one mapping per line. Each line should use the following format:

class_name,id
Indexing for classes starts at 0. Do not include a background class as it is implicit.

For example:

cow,0
cat,1
bird,2

# Debugging
Creating your own dataset does not always work out of the box. There is a debug.py tool to help find the most common mistakes.

Particularly helpful is the --annotations flag which displays your annotations on the images from your dataset. Annotations are colored in green when there are anchors available and colored in red when there are no anchors available. If an annotation doesn't have anchors available, it means it won't contribute to training. It is normal for a small amount of annotations to show up in red, but if most or all annotations are red there is cause for concern. The most common issues are that the annotations are too small or too oddly shaped (stretched out)."


