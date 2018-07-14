import os
from os import listdir
from os.path import isfile, join, isdir
import cv2
from action_dictionary import action_dictionary
import numpy as np


def loadImgs(preprocess_dir=join(os.getcwd(), "preprocess")):
    print("load from:"+preprocess_dir)

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    i = 0
    for action in action_dictionary:
        dir_path = join(preprocess_dir, action)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        print("loading training data from: "+dir_path, end="")
        imgs = list(join(dir_path, img) for img in listdir(dir_path))
        imgs = list(img for img in imgs if isfile(img))
        print(" -> {0} imgs".format(len(imgs)))
        for img in imgs:
                x_train.append(cv2.imread(img, 0))
                y_train.append(i)
        i += 1
    return (x_train, y_train)
