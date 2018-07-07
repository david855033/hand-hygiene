import cv2
import os
from os.path import join
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
from keras.models import load_model
from dataloader_module import loadImgs
from tensorflow.python.client import device_lib


def train(source_path=join(os.getcwd(), "preprocess"),
          model_save_name="model",
          model_load_name="model"):

    (x_train, y_train, categories) = loadImgs(source_path)

    input_shape = (256, 256, 1)
    batch_size = 32
    epochs = 12
    num_classes = len(categories)

    x_train = np.array(x_train).astype('float32')
    x_train /= 255
    x_train = x_train.reshape(x_train.shape[0], 256, 256, 1)
    print(x_train.shape)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    print(y_train.shape)

    model = object()

    print("======device information========")
    print(device_lib.list_local_devices())
    print("======device information========")

    if not os.path.exists(join(os.getcwd(), 'models')):
        os.makedirs(join(os.getcwd(), 'models'))

    model_path = join(os.getcwd(), 'models', model_load_name+".h5")
    if model_load_name != "" and os.path.isfile(model_path):
        print('load model from: ' + model_path)
        model = load_model(model_path)
    else:
        print('create new model: ' + model_path)
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              callbacks=[TensorBoard(log_dir='./tmp/')])

    if model_save_name != "":
        model.save(join(os.getcwd(), 'models', model_save_name+".h5"),
                   overwrite=True)

# shot tensorboard=>  tensorboard --logdir=./tmp
