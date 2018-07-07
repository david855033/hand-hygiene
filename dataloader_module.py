import os
from os import listdir
from os.path import isfile, join, isdir
import cv2


def loadImgs(preprocess_dir=join(os.getcwd(), "preprocess")):
    print("load from:"+preprocess_dir)
    categories = list(d for d in listdir(preprocess_dir)
                      if isdir(join(preprocess_dir, d)))
    x_train = []
    y_train = []
    i = 0
    for c in categories:
        dir_path = join(preprocess_dir, c)
        print("Indexing imgs from: "+dir_path)
        imgs = list(join(dir_path, i) for i in listdir(dir_path))
        imgs = list(i for i in imgs if isfile(i))
        print(" -> {0} imgs were loaded".format(len(imgs)))
        for img in imgs:
            x_train.append(cv2.imread(img, 0))
            y_train.append(i)
        i += 1
    return (x_train, y_train, categories)
