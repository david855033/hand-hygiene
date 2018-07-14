import cv2
import os
from keras.models import load_model
from action_dictionary import action_dictionary


def predict(modelsrc, image):
    model = load_model(modelsrc)
    result = model.predict(image)

    result = result.flatten()
    i = 0
    resultText = []

    for r in result.astype('str'):
        resultText.append(action_dictionary[i]+":"+r)
        i += 1
    resultText.append("prediction: "+action_dictionary[result.argmax(axis=0)])
    return resultText
