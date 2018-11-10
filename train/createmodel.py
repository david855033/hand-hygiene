import numpy as np
np.random.seed(1337)
import keras
from keras import regularizers
from keras.applications.mobilenetv2 import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import layers
from keras.models import load_model
from tensorflow.python.client import device_lib
from share.global_setting import ACTIONS


def create_model():
    input_shape = (224, 224, 3)
    num_classes = len(ACTIONS)
    alpha = 0.35
    base_model = MobileNetV2(
        input_shape=input_shape, alpha=alpha, depth_multiplier=1,
        include_top=False, weights='imagenet', input_tensor=None, pooling='avg')

    # for layer in base_model.layers[:147]:
    #     layer.trainable = False

    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer('block_9_add').output)

    x = model.output

    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                            expansion=6, block_id=10)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                            expansion=6, block_id=11)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                            expansion=6, block_id=12)

    # x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
    #                         expansion=6, block_id=13)
    # x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
    #                         expansion=6, block_id=14)
    # x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
    #                         expansion=6, block_id=15)
    # x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
    #                         expansion=6, block_id=16)

    x = layers.Conv2D(24,
                      kernel_size=1,
                      use_bias=False,
                      name='Conv_1')(x)
    x = layers.BatchNormalization(epsilon=1e-3,
                                  momentum=0.999,
                                  name='Conv_1_bn')(x)
    x = layers.ReLU(6., name='out_relu')(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = Dense(24, use_bias=True, activation='relu',
              kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, use_bias=True, activation='softmax',
              kernel_regularizer=regularizers.l2(0.01))(x)
    model = Model(input=model.input, output=x)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.0001),
                  metrics=['accuracy'])

    keras.utils.print_summary(model)
    return model


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    import keras.backend as backend

    in_channels = backend.int_shape(inputs)[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}_'.format(block_id)

    if block_id:
        # Expand
        x = layers.Conv2D(expansion * in_channels,
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          activation=None,
                          name=prefix + 'expand')(x)
        x = layers.BatchNormalization(epsilon=1e-3,
                                      momentum=0.999,
                                      name=prefix + 'expand_BN')(x)
        x = layers.ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    # Depthwise
    if stride == 2:
        x = layers.ZeroPadding2D(padding=correct_pad(backend, x, 3),
                                 name=prefix + 'pad')(x)
    x = layers.DepthwiseConv2D(kernel_size=3,
                               strides=stride,
                               activation=None,
                               use_bias=False,
                               padding='same' if stride == 1 else 'valid',
                               name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'depthwise_BN')(x)

    x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = layers.Conv2D(pointwise_filters,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      activation=None,
                      name=prefix + 'project')(x)
    x = layers.BatchNormalization(
        epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')(x)

    if in_channels == pointwise_filters and stride == 1:
        return layers.Add(name=prefix + 'add')([inputs, x])
    return x


def correct_pad(backend, inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.
    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.
    # Returns
        A tuple.
    """
    img_dim = 2 if backend.image_data_format() == 'channels_first' else 1
    input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))
