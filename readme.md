# Hand-Hygiene

## extract.py:
extract images from assigned video clip
### usage:
```
python extract.py .\videoname.mp4 [-f folder_path] [-n max_img_number] [-r extract_ratio]
```
note: <br>
**folder_path**: set the folder path, (default = ./dest)<br>
**max_img_number**: limit the maximum of images been extracted in one video <br>
**extract_ratio**: how many frames should be read/skipped before capture one frame img  <br>