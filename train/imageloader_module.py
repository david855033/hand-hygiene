import cv2
import os
from os.path import isdir, isfile, join, splitext


def loadImage(img_source):
    if not os.path.exists(img_source):
        return
    filename, file_extention = splitext(img_source)
    if file_extention.lower() != ".jpg":
        return
    return cv2.imread(img_source)
