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
    epochs = 6
    num_classes = len(ACTIONS)
    model = create_model()

    x_train, y_train = loadImgs(source_path)

    x_train = np.array(x_train).astype('float32')
    x_train /= 255
    x_train = x_train.reshape(x_train.shape[0], 128, 128, 1)
    y_train = keras.utils.to_categorical(y_train, num_classes)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_split=0.2,
              callbacks=[TensorBoard(log_dir='./tmp/')])

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
