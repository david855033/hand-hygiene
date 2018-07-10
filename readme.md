# Hand-Hygiene
## extractfolder.py
### description
Extract images from all video in "./videosrc" folder,</br>
then sort the result images into one of the following 7 categories:</br>
 'palm', 'handback', 'finger', 'hook', 'thumb', 'tip', 'wrist'.
The destiny folder is ./dest/[category_name].
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

### usage
```
python trainmodel.py [-f folder containing training (default:./preprocess)[-s name_to_save_model] [-l name_to_load_model]
```
will automatic load/save with overwrite to ./model/model.h5 unless specifying model name

### tensorboard
use the following command to show tensorboard during training
```
tensorboard --logdir=./tmp
```