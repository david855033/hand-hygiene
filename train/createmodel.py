import keras
from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
from tensorflow.python.client import device_lib
from share.global_setting import ACTIONS


def create_model():
    input_shape = (224, 224, 3)
    num_classes = len(ACTIONS)

    base_model = VGG19(weights='imagenet', include_top=False,
                       input_shape=input_shape)

    for layer in base_model.layers:
        layer.trainable = False
    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer('block4_pool').output)
    x = model.output
    x = BatchNormalization()(x)
    x = Conv2D(filters=24, kernel_size=(1, 1),
               activation='relu', padding='same',
               use_bias=False)(x)
    x = Conv2D(filters=24, kernel_size=(3, 3),
               activation='relu', padding='same')(x)
    x = Conv2D(filters=24, kernel_size=(3, 3),
               activation='relu', padding='same')(x)
    x = Conv2D(filters=24, kernel_size=(3, 3),
               activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(24, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(input=model.input, output=x)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    keras.utils.print_summary(model)
    return model
