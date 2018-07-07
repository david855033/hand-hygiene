# Hand-Hygiene

## extract.py:
extract images from assigned video clip
### usage:
```
python extract.py .\videoname.mp4 [-f folder_path] [-n max_img_number] [-r extract_ratio]
```
### Note: 
**folder_path**<br>
set the destination folder path, to which all image will be saved (default = ./dest)<br>
**max_img_number**<br>
limit the maximum of images to be extracted in one video <br>
**extract_ratio**<br>
indicate how many frames should be read/skipped before capture one frame<br>

## extractfolder.py
extract images from all video in src folder
### usage
```
python extractfolder.py [-s srcfolder (default='.\videosrc')] [-f destfolder (default='.\dest')] [-n max_img_number] [-r extract_ratio]
```
### Note: 
**srcfolder**<br>
folder contain source videos<br>
**destfolder**<br>
output folder, output image will be separete into different folders according to original video filename
## imgpreprocess.py
### usage
```
python imagepreprocess.py [-s srcfolder (default='.\dest')] [-p preprocess_path (default='.\preprocess')]
```
# trainmodel.py
### description
    **use folders containing classified preprocessed data to generate a keras model(classifier)**
### usage
```
python trainmodel.py [-f folder containing training imgs][-s name_to_save_model] [-l name_to_load_model]
```
will automatic load/save with overwrite to ./model/model.h5 unless specifying model name

### tensorboard
use the following command to show tensorboard during training
```
tensorboard --logdir=./tmp
```