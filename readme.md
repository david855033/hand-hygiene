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