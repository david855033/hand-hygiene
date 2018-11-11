import numpy as np
import math
import pandas as pd
import os
from os.path import exists, join, isfile, dirname, split
from os import listdir
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import cv2

import random
import keras
from keras.models import load_model, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils.generic_utils import CustomObjectScope


from keras import backend as K
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

from share.global_setting import ACTIONS


def testmodel(kfold):
    random.seed(10)
    group_order = get_random_group_order(20)
    print(group_order)

    for fold in range(kfold):
        x_test, y_test = load_imgs_by_group(
            source_path=r".\data\preprocess", group_order=group_order,
            group_count=20, kfold=kfold, fold=fold,
            img_per_class_person=10)

        print(x_test.shape, y_test.shape)

        modelpath = "D:\hand-hygiene\data\model\model_fold{0}.h5".format(fold)
        model = getModel(modelpath)

        prediction = model.predict(x_test)

        y_decode = decode(y_test).tolist()
        prediction_decode = decode(prediction).tolist()

        print(y_decode)
        print(prediction_decode)

        data = pd.DataFrame(
            {"gt": y_decode, "predict": prediction_decode},
            index=list(range(len(prediction))))
        data.to_csv(r".\data\test\fold{0}.csv".format(fold))


def decode(datum):
    return np.argmax(datum, axis=1)


def load_imgs_by_group(
    source_path, group_order, group_count, kfold, fold,
        img_per_class_person):
    x_test, y_test = [], []
    filename_set = {}

    val_group_count = math.floor(group_count/kfold)
    startPos = val_group_count*fold
    endPos = startPos+val_group_count
    val_groups = group_order[startPos:endPos]
    print('fold={0}, valgroup={1}'.format(fold, val_groups))
    for action_index, action in enumerate(ACTIONS):
        action_folder_path = join(source_path, action)
        filename_set[action] = {}

        for filename in listdir(action_folder_path):
            path = join(action_folder_path, filename)
            index = int(filename.split('_')[0])
            if index in val_groups or action == "background":
                if index not in filename_set[action]:
                    filename_set[action][index] = []
                filename_set[action][index].append(path)

        estimate_img_per_class = img_per_class_person * val_group_count
        estimate_img_per_index = math.floor(
            estimate_img_per_class / len(filename_set[action]))

        if action == "background":
            for filenames in filename_set[action].values():
                random.shuffle(filenames)
                pos = 0
                for i in range(estimate_img_per_index):
                    x_test.append(read_img(filenames[pos]))
                    y_test.append(0)
                    pos = pos + 1 if pos < len(filenames)-1 else 0
        else:
            available_val_groups = []
            for val_group in val_groups:
                if val_group in filename_set[action]:
                    available_val_groups.append(val_group)

            corrected_img_per_index = math.floor(
                estimate_img_per_index * group_count / kfold /
                len(available_val_groups))

            for val_group in available_val_groups:
                filenames = filename_set[action][val_group]
                random.shuffle(filenames)
                pos = 0
                for i in range(corrected_img_per_index):
                    x_test.append(read_img(filenames[pos]))
                    y_test.append(action_index)
                    pos = pos + 1 if pos < len(filenames)-1 else 0

        print("{0}:{1}".format(action, len(filename_set[action])))

    x_test = np.array(x_test, dtype='float32')
    y_test = np.array(y_test, dtype='float32')
    y_test = keras.utils.to_categorical(y_test, len(ACTIONS))
    return x_test, y_test


def read_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32')
    img = keras.applications.mobilenet.preprocess_input(img)
    return img


def getModel(model_save_path=""):
    model = object()
    if exists(model_save_path):
        print('"{0}"  exsists, resuming'.format(model_save_path))
        with CustomObjectScope({'relu6': ReLUs(keras.layers.ReLU(6.)),
                                'DepthwiseConv2D':
                                keras.layers.DepthwiseConv2D}):
            model = load_model(model_save_path)
        # for layer in model.layers[:-5]:
        #     layer.trainable = False
        # model = Model(model.inputs, model.outputs)
        # keras.utils.print_summary(model)
    else:
        model = create_model()
    return model


class ReLUs(Activation):
    def __init__(self, activation, **kwargs):
        super(ReLUs, self).__init__(activation, **kwargs)
        self.__name__ = 'relu6'


def get_random_group_order(group_count=20):
    groups_order = list(range(1, group_count+1))
    random.shuffle(groups_order)
    return groups_order


testmodel(5)
