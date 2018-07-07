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
### usage
```
python trainmodel.py [-m dirpath_to_save_model] [-s dirpath_of_source_preprocessed_img]
```
not done yet, should generate a keras model base on the folder containing preprocessed data
### tensorboard
use the following command to show tensorboard during training
```
tensorboard --logdir=./tmp
```