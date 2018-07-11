# Hand-Hygiene
## extractfolder.py
### description
Extract images from all video in "./videosrc" folder,</br>
then sort the result images into one of the following 7 categories:</br>
 'palm', 'handback', 'finger', 'hook', 'thumb', 'tip', 'wrist'.
The destiny folder is ./dest/[category_name].</br>
**NOTE: the filename of the video must contain a category name!**
### usage
```
python extractfolder.py [-s srcfolder (default='.\videosrc')] [-f destfolder (default='.\dest')] [-n max_img_number="default=0(no restriction)"] [-r extract_ratio(default=5)]
```
## imgpreprocess.py
### description
Load images from ".dest", and do default image preprocessing:</br>
**resize to 128x128 and covert to greyscale**</br>
then save to './preprocess' with original path structure remained.
### usage
```
python imagepreprocess.py [-s sourcefolder[default=./dest]] [-p preprocess[default:./preprocess]]
```

# trainmodel.py
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