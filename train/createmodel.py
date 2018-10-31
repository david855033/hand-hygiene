import keras
from keras.applications.mobilenetv2 import MobileNetV2
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

    base_model = MobileNetV2(input_shape=input_shape, alpha=1.0, depth_multiplier=1, include_top=False,
                             weights='imagenet', input_tensor=None, pooling='avg')

    for layer in base_model.layers[:-56]:
        layer.trainable = False
    model = Model(inputs=base_model.input,
                  outputs=base_model.output)

    x = model.output
    # x = Conv2D(filters=32, kernel_size=(1, 1), activation='relu',
    #            padding='same', use_bias=False)(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Flatten()(x)
    # x = Dropout(0.5)(x)
    x = Dense(num_classes, use_bias=True, activation='softmax')(x)
    model = Model(input=model.input, output=x)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    keras.utils.print_summary(model)
    return model
