import numpy as np
from os.path import exists, join, isfile, dirname
from os import listdir
from share.global_setting import ACTIONS
from share.file_util import checkfolder
import cv2
from train.createmodel import create_model
from keras.models import load_model
from keras.callbacks import TensorBoard
import keras
import random


def trainmodel(source_path, model_save_path):
    print('\n[train model]')
    print(' >> traning data src: {0}'.format(source_path))
    print(' >> model destination: {0}'.format(model_save_path))

    checkfolder(dirname(model_save_path))

    model = object()
    if exists(model_save_path):
        input('"{0}"'
              .format(model_save_path) +
              " exsists, press Enter to overwrite. " +
              "Press Ctrl+C and Enter to Abort.")
    batch_size = 32
    epochs = 12
    num_classes = len(ACTIONS)
    model = create_model()

    x_data, y_data = loadImgs(source_path)

    datacount = len(x_data)
    traincount = int(datacount * 0.7)
    valcount = int(datacount * 0.2)
    testcount = int(datacount * 0.1)

    ind_list = [i for i in range(datacount)]
    random.shuffle(ind_list)
    x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []

    for i in range(traincount):
        pos = ind_list[i]
        x_train.append(x_data[pos])
        y_train.append(y_data[pos])
    for i in range(traincount, traincount + valcount):
        pos = ind_list[i]
        x_val.append(x_data[pos])
        y_val.append(y_data[pos])
    for i in range(traincount + valcount, len(x_data)):
        pos = ind_list[i]
        x_test.append(x_data[pos])
        y_test.append(y_data[pos])

    x_train = np.array(x_train).astype('float32')
    x_train /= 255
    x_train = x_train.reshape(x_train.shape[0], 128, 128, 1)
    y_train = keras.utils.to_categorical(y_train, num_classes)

    x_val = np.array(x_val).astype('float32')
    x_val /= 255
    x_val = x_val.reshape(x_val.shape[0], 128, 128, 1)
    y_val = keras.utils.to_categorical(y_val, num_classes)

    x_test = np.array(x_test).astype('float32')
    x_test /= 255
    x_test = x_test.reshape(x_test.shape[0], 128, 128, 1)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print("train N={0}, val N={1}, test N={2}".format(
        traincount, valcount, testcount))
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_val, y_val),
              callbacks=[TensorBoard(log_dir='./tmp/')])

    model.test(x_train, y_train)
    model.save(model_save_path, overwrite=True)


def loadImgs(source_path):
    x_train, y_train = [], []
    count = 0
    i = 0
    for action in ACTIONS:
        action_folder_path = join(source_path, action)
        imgs = list(join(action_folder_path, img)
                    for img in listdir(action_folder_path))
        for img in imgs:
            x_train.append(cv2.imread(img, 0))
            y_train.append(i)
            count += 1
        i += 1
        print(" >> Loading {0} imgs from:".format(len(imgs)) +
              action_folder_path)
    print(" >> total: {0} labeled imgs".format(count))
    return x_train, y_train
