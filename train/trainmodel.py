import numpy as np
import os
from os.path import exists, join, isfile, dirname, split
from os import listdir
from share.global_setting import ACTIONS
from share.file_util import checkfolder
import cv2
from train.createmodel import create_model
from keras.models import load_model, Model
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import keras
import random


def trainmodel(source_path, model_save_path, model_autosave_path):
    print('\n[train model]')
    print(' >> traning data src: {0}'.format(source_path))
    print(' >> model destination: {0}'.format(model_save_path))

    model = object()
    if exists(model_save_path):
        input('"{0}"'
              .format(model_save_path) +
              " exsists, press Enter to overwrite. " +
              "Press Ctrl+C and Enter to Abort.")
    model = create_model()

    checkfolder(dirname(model_save_path))
    augment_path = r'./data/augment'
    checkfolder(augment_path)
    for the_file in os.listdir(augment_path):
        file_path = os.path.join(augment_path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

    batch_size = 128
    epochs = 300
    num_classes = len(ACTIONS)

    x_train, y_train, x_val, y_val, x_test, y_test = loadImgs(source_path)
    x_train = np.array(x_train).astype('float32')
    x_train = x_train.reshape(x_train.shape[0], 224, 224, 3)
    y_train = keras.utils.to_categorical(y_train, num_classes)

    x_val = np.array(x_val).astype('float32')
    x_val = x_val.reshape(x_val.shape[0], 224, 224, 3)
    y_val = keras.utils.to_categorical(y_val, num_classes)

    x_test = np.array(x_test).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 224, 224, 3)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print("train N={0}, val N={1}, test N={2}".format(
        len(x_train), len(x_val), len(x_test)))

    datagen = ImageDataGenerator(
        preprocessing_function=keras.applications.mobilenet.preprocess_input,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        rotation_range=20,
        shear_range=10
    )

    val_batches = ImageDataGenerator(
        preprocessing_function=keras.applications.mobilenet.preprocess_input)

    i = 0
    preview_img = x_train[0]
    preview_img = img_to_array(preview_img)
    preview_img = preview_img.reshape((1,) + preview_img.shape)
    # save some preview img of augmented data
    for batch in datagen.flow(
            preview_img,
            batch_size=1, save_to_dir='data/augment', save_prefix='preview',
            save_format='jpeg'):
        i += 1
        if i > 20:
            break

    model.fit_generator(
        datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True),
        steps_per_epoch=int(len(x_train) / batch_size),
        epochs=epochs, verbose=1,
        validation_data=val_batches.flow(
            x_val, y_val, batch_size=batch_size, shuffle=True),
        callbacks=[TensorBoard(log_dir='./tmp/set16_mnv2_run10_train56layer_alpha1.0_no validation preprocess'),
                   keras.callbacks.ModelCheckpoint(
                       model_autosave_path, monitor='val_acc', mode='max',
                       verbose=1, save_best_only=True, period=5)])

    model.save(model_save_path, overwrite=True)
    if len(x_test) > 0:
        score = model.evaluate(x_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def showPreview(x_train, y_train, x_val, y_val):
    #todo
    # keras.applications.mobilenet.preprocess_input
    print('preview')
    return


def loadImgs(source_path, setcount=16, valcount=3, testcount=0):
    x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []
    count = 0
    i = 0
    set_index = list(range(1, setcount + 1))
    random.shuffle(set_index)
    traincount = setcount - testcount - valcount

    trainset = set_index[0:traincount]
    valset = set_index[traincount:traincount+valcount]
    testset = set_index[traincount+valcount:traincount+valcount+testcount]

    for action in ACTIONS:
        action_folder_path = join(source_path, action)
        imgs = list(join(action_folder_path, img)
                    for img in listdir(action_folder_path))
        for img in imgs:
            if int(split(img.split("_")[0])[1]) in testset:
                toAppend = cv2.imread(img)
                toAppend = cv2.cvtColor(toAppend, cv2.COLOR_BGR2RGB)
                x_test.append(toAppend)
                y_test.append(i)
            elif int(split(img.split("_")[0])[1]) in valset:
                toAppend = cv2.imread(img)
                toAppend = cv2.cvtColor(toAppend, cv2.COLOR_BGR2RGB)
                x_val.append(toAppend)
                y_val.append(i)
            elif int(split(img.split("_")[0])[1]) in trainset:
                toAppend = cv2.imread(img)
                toAppend = cv2.cvtColor(toAppend, cv2.COLOR_BGR2RGB)
                x_train.append(toAppend)
                y_train.append(i)
            count += 1
        i += 1
        print(" >> Loading {0} imgs from:".format(len(imgs)) +
              action_folder_path)
    print(" >> total: {0} labeled imgs".format(count))
    print(" >> dataset count={0}, val set={1},test set={2},".format(
        setcount, valset, testset))
    return x_train, y_train, x_val, y_val, x_test, y_test
