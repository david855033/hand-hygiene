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
from dataloader_module import loadImgs


def train(source_path=join(os.getcwd(), "preprocess")):
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

# shot tensorboard=>  tensorboard --logdir=./tmp
