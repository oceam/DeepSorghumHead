# README: DeepSorghumHead

This is a support page of paper published at "Plant Phenomics" as an Original Research. 

> Ghosal,S., Zheng, B., Chapman, S.C., Potgieter, A.B., Jordan,D.R., Wang,X., Singh, A.K., Singh, A., Hirafuji, M., Ninomiya, S., Ganapathysubramanian, B., Sarkar, S., Guo, W.(2019, under review). A weakly supervised deep learning framework for sorghum head detection and counting. Plant Phenomics.

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

# TRAINED MODEL located in DeepSorghumHead_Keras/bin/inf_models/

# Running the ToolChain: (navigate to folder where your .py file is located first, open a terminal window and execute following syntax)

1. For Training (from scratch or using pre-trained model):
<<<<<<< HEAD
<<<<<<< HEAD
CUDA_VISIBLE_DEVICES=<### ENTER GPU DEVICE ID ###> python train_custom.py --backbone resnet50 --weights <### ENTER PATH TO PRE_TRAINED MODEL WEIGHTS ###> --batch-size 10 --epochs 200 --steps <### NUMBER OF STEPS - THIS IS DETERMINED BY YOUR TRAINING DATA SIZE AND BATCH SIZE, STEPS = TRAINING DATA SIZE/BATCH SIZE ###> --snapshot-path <### ENTER PATH TO SAVE NEW MODEL WEIGHTS ###> --tensorboard-dir <### ENTER TENSORBOARD LOG PATH ###> --random-transform --image-min-side 1600 --image-max-side 2400 csv <### ENTER PATH TO TRAINING DATA (.csv FILE) ###> <### ENTER PATH TO ANNOTATION MAP (THE 'annotation_map.csv' FILE) ###>

2. For converting your Trained model to an Inference Model (this must be done before you can test your model on test data):
python convert_model.py <### ENTER PATH TO PRE_TRAINED MODEL WEIGHTS ###> <### ENTER PATH TO SAVE CONVERTED MODEL WEIGHTS ###>

3. Evaluate the Trained Model on Test Data:
=======
=======
>>>>>>> 2c1cd4bda91a661fd5e375baf8cfc68a12b27afe

CUDA_VISIBLE_DEVICES=<### ENTER GPU DEVICE ID ###> python train_custom.py --backbone resnet50 --weights <### ENTER PATH TO PRE_TRAINED MODEL WEIGHTS ###> --batch-size 10 --epochs 200 --steps <### NUMBER OF STEPS - THIS IS DETERMINED BY YOUR TRAINING DATA SIZE AND BATCH SIZE, STEPS = TRAINING DATA SIZE/BATCH SIZE ###> --snapshot-path <### ENTER PATH TO SAVE NEW MODEL WEIGHTS ###> --tensorboard-dir <### ENTER TENSORBOARD LOG PATH ###> --random-transform --image-min-side 1600 --image-max-side 2400 csv <### ENTER PATH TO TRAINING DATA (.csv FILE) ###> <### ENTER PATH TO ANNOTATION MAP (THE 'annotation_map.csv' FILE) ###>

2. For converting your Trained model to an Inference Model (this must be done before you can test your model on test data):

python convert_model.py <### ENTER PATH TO PRE_TRAINED MODEL WEIGHTS ###> <### ENTER PATH TO SAVE CONVERTED MODEL WEIGHTS ###>

3. Evaluate the Trained Model on Test Data:

<<<<<<< HEAD
>>>>>>> 2c1cd4bda91a661fd5e375baf8cfc68a12b27afe
=======
>>>>>>> 2c1cd4bda91a661fd5e375baf8cfc68a12b27afe
python evaluate.py --backbone resnet50 --max-detections 600 --save-path <### ENTER PATH FOR OUTPUT DIRECTORY ###> csv <### ENTER PATH WHERE TEST DATA IS LOCATED (.csv FILE) ###> <### ENTER PATH TO ANNOTATION MAP (THE 'annotation_map.csv' FILE) ###> <### ENTER PATH WHERE TRAINED INFERENCE MODEL IS LOCATED ###>

# Base Code originally from https://github.com/fizyr/keras-retinanet


