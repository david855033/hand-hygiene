# Hand-Hygiene Project
We are building a real-time, 24 hours a day monitoring system which gives instant feedback on hand-washing process.
This system is a proof of concept powered by deep learning-based computer vision technology and is feasible to implement in clinical setting.

## Data pipline
**all data are put in the folder: ./data**
```
src video: ./data/videosrc
-> extract.py
-> extracted frame: ./data/extract
-> preprocess.py
-> preprocessed frame: ./data/preprocess
-> train.py
-> generated Keras model: ./data/model
```

## extract.py
### description
Extract images from videos in "./data/videosrc" folder,</br>
then classify the images into one of the following 7 categories:</br>
 'palm', 'handback', 'finger', 'hook', 'thumb', 'tip', 'wrist' according to the video filename.
The destiny folder is ./data/extract/[category_name].</br>
**NOTE: the filename of the video must contain a category name!**
### usage
```
python extract.py
```
## preprocess.py
### description
Load images from "./data/extract", and perform image preprocessing:</br>
**resize to 128x128 and covert to greyscale.**</br>
Save preprocessed images to './data/preprocess'
### usage
```
python preprocess.py
```

## train.py
### description
load image from "./preprocess"</br>
the images must have be sorted in to default 7 categories by folders.</br>
train a model for image classification task</br>
then save the model to './models/model.h5'.</br>
By default, if the model exists, this trainer will load the exsisted model to continue weight training, then overwrite it after training. 
### usage
```
python trainmodel.py [-f folder(default:'./preprocess')[-s name_to_save_model] [-l name_to_load_model]
```
### tensorboard
use the following command to show tensorboard during training
```
tensorboard --logdir=./tmp
```