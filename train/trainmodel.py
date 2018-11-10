import numpy as np
import math
import os
import tensorflow as tf
from os.path import exists, join, isfile, dirname, split
from os import listdir
from share.global_setting import ACTIONS
from share.file_util import checkfolder
import cv2

from keras.layers import Activation
from train.createmodel import create_model
from keras.models import load_model, Model
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.utils import multi_gpu_model
import keras
import random
from keras.utils.generic_utils import CustomObjectScope


def trainmodel(source_path=r".\data\preprocess",
               model_save_folder=r".\data\model"):
    print('\n[train model]')
    print(' >> traning data src: {0}'.format(source_path))
    print(' >> model destination: {0}'.format(model_save_folder))

    checkfolder(dirname(model_save_folder))
    augment_path = r'./data/augment'
    checkfolder(augment_path)

    clear_augment_dir(augment_path)

    kfold = 0  # 5
    batch_size = 128
    epochs = 140
    num_classes = len(ACTIONS)
    initial_epoch = 0
    group_count = 20
    random.seed(10)
    group_order = get_random_group_order(group_count)
    print("group order: ", group_order)
    img_per_class_person = 100
    dataset_img = load_imgs(source_path, max=-1)

    if kfold == 0:
        print('no validation!')
        x_train, y_train = get_no_validation(
            dataset_img=dataset_img, num_classes=num_classes,
            group_count=group_count,
            img_per_class_person=img_per_class_person)

        print(x_train.shape, y_train.shape)
        model_save_path = model_save_folder + r"\model_no_validation.h5"
        
        model = getModel(model_save_path)

        datagen_train = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            rotation_range=20,
            shear_range=10
        )

        model.fit_generator(
            datagen_train.flow(
                x_train, y_train, batch_size=batch_size, shuffle=True),
            steps_per_epoch=int(len(x_train) / batch_size),
            epochs=epochs, verbose=1,
            initial_epoch=initial_epoch,
            callbacks=[
                TensorBoard(
                    log_dir='./tmp/run39_no_valid')])

        model.save(model_save_path, overwrite=True)

    else:
        for fold in range(0, kfold):
            print('fold:{0}/{1}'.format(fold, kfold))
            x_train, y_train, x_val, y_val = get_fold(
                group_order=group_order, dataset_img=dataset_img, fold=fold,
                kfold=kfold, num_classes=num_classes, group_count=group_count,
                img_per_class_person=img_per_class_person)

            model_save_path = model_save_folder + \
                r"\model_fold{0}.h5".format(fold)
            model_training_path = model_save_folder + \
                r"\model_training_fold{0}.h5".format(fold)

            model = getModel(model_save_path)

            # parallel_model = multi_gpu_model(model, cpu_merge=False)
            # parallel_model.compile(loss=keras.losses.categorical_crossentropy,
            #                        optimizer=keras.optimizers.Adam(),
            #                        metrics=['accuracy'])

            datagen_train = ImageDataGenerator(
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                rotation_range=20,
                shear_range=10
            )

            model.fit_generator(
                datagen_train.flow(
                    x_train, y_train, batch_size=batch_size, shuffle=True),
                steps_per_epoch=int(len(x_train) / batch_size),
                epochs=epochs, verbose=1,
                validation_data=(x_val, y_val),
                initial_epoch=initial_epoch,
                callbacks=[
                    TensorBoard(
                        log_dir='./tmp/run38_nofreeze_layer10-12_filter24-24_fold_to140_{0}'.format(fold)),
                    keras.callbacks.ModelCheckpoint(
                        model_training_path, monitor='val_loss', mode='auto',
                        verbose=1, save_best_only=True, period=5)])

            model.save(model_save_path, overwrite=True)
    return

# i = 0
# preview_img = x_train[random.randint(0, len(x_train)-1)]
# preview_img = img_to_array(preview_img)
# preview_img = preview_img.reshape((1,) + preview_img.shape)
# # save some preview img of augmented data
# for batch in datagen.flow(
#         preview_img,
#         batch_size=1, save_to_dir='data/augment', save_prefix='preview',
#         save_format='jpeg'):
#     i += 1
#     if i > 20:
#         break


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


def clear_augment_dir(augment_path):
    for the_file in os.listdir(augment_path):
        file_path = os.path.join(augment_path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)
    return


def showPreview(x_train, y_train, x_val, y_val):
    # todo
    # keras.applications.mobilenet.preprocess_input
    print('preview')
    return


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


def get_random_group_order(group_count=20):
    groups_order = list(range(1, group_count+1))
    random.shuffle(groups_order)
    return groups_order


def get_no_validation(dataset_img, num_classes, group_count,
                      img_per_class_person):
    x_train, y_train = [], []

    background_imgs = []
    for img_group in dataset_img[ACTIONS[0]].values():
        background_imgs.extend(img_group)

    pos = 0
    for i in range(img_per_class_person*group_count):
        x_train.append(background_imgs[pos])
        y_train.append(0)
        pos = pos + 1 if pos < len(background_imgs)-1 else 0

    for i in range(1, num_classes):
        action = ACTIONS[i]
        # train data
        availible_train_group = []
        for g in range(1, group_count+1):
            if g in dataset_img[action]:
                availible_train_group.append(g)

        correct_img_per_class_person = math.floor(
            img_per_class_person / len(availible_train_group) *
            group_count)

        for g in availible_train_group:
            pos = 0
            for j in range(correct_img_per_class_person):
                x_train.append(dataset_img[action][g][pos])
                y_train.append(i)
                pos = pos + 1 if pos < len(dataset_img[action][g])-1 else 0

    x_train = np.array(x_train, dtype='float32')
    y_train = np.array(y_train, dtype='float32')
    y_train = keras.utils.to_categorical(y_train, num_classes)
    return x_train, y_train


def get_fold(group_order, dataset_img, fold,
             kfold, group_count, num_classes,
             img_per_class_person):
    x_train, y_train, x_val, y_val = [], [], [], []

    val_group_count = math.floor(group_count/kfold)
    train_group_count = group_count-val_group_count
    startPos = val_group_count*fold
    endPos = startPos+val_group_count
    val_groups = group_order[startPos:endPos]
    train_groups = group_order[0:startPos] + group_order[endPos:]
    print('val_group = {0}'.format(val_groups))
    # fill backgroun img into train/val
    background_imgs = []
    for img_group in dataset_img[ACTIONS[0]].values():
        background_imgs.extend(img_group)
    val_background_count = math.floor(len(background_imgs) / kfold)

    train_background_img = background_imgs[val_background_count:]
    pos = 0
    for i in range(img_per_class_person*train_group_count):
        x_train.append(train_background_img[pos])
        y_train.append(0)
        pos = pos + 1 if pos < len(train_background_img)-1 else 0

    val_background_img = background_imgs[0:val_background_count]
    pos = 0
    for i in range(img_per_class_person*val_group_count):
        x_val.append(train_background_img[pos])
        y_val.append(0)
        pos = pos + 1 if pos < len(val_background_img)-1 else 0

    # fill actions img into train/val
    for i in range(1, num_classes):
        action = ACTIONS[i]
        # train data
        availible_train_group = []
        for g in train_groups:
            if g in dataset_img[action]:
                availible_train_group.append(g)

        correct_img_per_class_person = math.floor(
            img_per_class_person / len(availible_train_group) *
            len(train_groups))

        for g in availible_train_group:
            pos = 0
            for j in range(correct_img_per_class_person):
                x_train.append(dataset_img[action][g][pos])
                y_train.append(i)
                pos = pos + 1 if pos < len(dataset_img[action][g])-1 else 0
        # validation data
        availible_val_group = []
        for g in val_groups:
            if g in dataset_img[action]:
                availible_val_group.append(g)

        correct_img_per_class_person = math.floor(
            img_per_class_person / len(availible_val_group) *
            len(val_groups))

        for g in availible_val_group:
            pos = 0
            for j in range(correct_img_per_class_person):
                x_val.append(dataset_img[action][g][pos])
                y_val.append(i)
                pos = pos + 1 if pos < len(dataset_img[action][g])-1 else 0

    x_train = np.array(x_train, dtype='float32')
    y_train = np.array(y_train, dtype='float32')
    y_train = keras.utils.to_categorical(y_train, num_classes)
    x_val = np.array(x_val, dtype='float32')
    y_val = np.array(y_val, dtype='float32')
    y_val = keras.utils.to_categorical(y_val, num_classes)

    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
    return x_train, y_train, x_val, y_val


trainmodel()
