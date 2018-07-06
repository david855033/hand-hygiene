import cv2
import os
from os.path import isdir, isfile, join


def imgPreprocess(source_path, preprocess_path):
    print('img preprocess{0}=>{1}'.format(source_path, preprocess_path))
    if not os.path.exists(preprocess_path):
        os.makedirs(preprocess_path)

    for root, dirs, files in os.walk(source_path):
        for name in files:
            if '.' not in name or name.split('.')[1].lower() != "jpg":
                continue
            img_from = join(root, name)
            root_to = root.replace(
                source_path, preprocess_path, 1)
            img_to = join(root_to, name)
            img = cv2.imread(img_from)
            if not os.path.exists(root_to):
                os.makedirs(root_to)
            # ---start convert
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (256, 256))
            # ---end convert
            cv2.imwrite(img_to, img)
            print("{0}->{1}".format(img_from, img_to))
