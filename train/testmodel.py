import numpy as np
from os.path import exists, join, isfile, dirname, split
import random
import keras
from keras.models import load_model, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils.generic_utils import CustomObjectScope


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from keras import backend as K
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

from share.global_setting import ACTIONS

def testmodel():
    random.seed(10)
    group_order = get_random_group_order(20)
    print(group_order)
    x_test, ytest = load_imgs_by_group(
        source_path=r".\data\preprocess",
        group_order=group_order, kfold=5, fold=0)

    with tf.device('cpu:0'):
        modelpath = r"D:\hand-hygiene\data\model\model_fold0.h5"
        model = getModel(modelpath)


def load_imgs(source_path, max=-1):
    dataset = {}
    for action in ACTIONS:
        print('loading image:{0}'.format(action))
        bin = {}
        action_folder_path = join(source_path, action)
        filenames = listdir(action_folder_path)
        random.shuffle(filenames)
        for filename in filenames:
            path = join(action_folder_path, filename)
            index = int(filename.split('_')[0])
            if index not in bin:
                bin[index] = []
            if max == -1 or len(bin[index]) < max:
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype('float32')
                img = keras.applications.mobilenet.preprocess_input(img)
                bin[index].append(img)
        dataset[action] = bin
        print('==>count: {0}'.format(len(bin)))
    return dataset


def load_imgs_by_group(source_path, group_order, kfold, fold):
    x_test, y_test = [], []
    background_folder_path = join(source_path, ACTIONS[0])
    filenames = listdir(background_folder_path)
    print(filenames)
    for i in range(1, len(ACTIONS)):
        action = ACTIONS[i]
        action_folder_path = join(source_path, action)
        filenames = listdir(action_folder_path)

    return x_test, y_test


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
        keras.utils.print_summary(model)
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


testmodel()
