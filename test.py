import random
import keras
from keras.models import load_model, Model
from train.createmodel import create_model
from keras.utils.generic_utils import CustomObjectScope

model = create_model()
model.save(r".\data\model\model_test.h5")
with CustomObjectScope({'relu6': keras.layers.ReLU(6.),
                        'DepthwiseConv2D':
                        keras.layers.DepthwiseConv2D}):
        model = load_model(r".\data\model\model_fold1.h5")


