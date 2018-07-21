import cv2
import os
import numpy as np
from os.path import join
from keras.models import load_model
from share.global_setting import ACTIONS
from preprocess.preprocess import preprocess_img
from random import shuffle


def predict_folder(model_path, img_folder):
    print('\n[Predict imgs in a folder]')
    print(' >> model_path: {0}'.format(model_path))
    print(' >> img_folder: {0}'.format(img_folder))

    model = load_model(model_path)
    img_list = []
    for root, dirs, files in os.walk(img_folder):
        for file in files:
            path = join(root, file)
            img_list.append(path)

    # 隨機選20張圖做測試
    shuffle(img_list)
    results = []
    for path in img_list[:20]:
        print(path+" "*20, end="\r")
        img = cv2.imread(path)
        preprocessed_img = preprocess_img(img)

        data = np.reshape(preprocessed_img, (1, 128, 128, 1))
        predict_result = model.predict(data)

        predict_result = predict_result.flatten()
        i = 0
        resultText = []
        for r in predict_result.astype('str'):
            resultText.append(ACTIONS[i]+":"+r)
            i += 1
        resultText.append(
            "prediction: "+ACTIONS[predict_result.argmax(axis=0)])

        preprocess_toshow = cv2.cvtColor(
            cv2.resize(preprocessed_img, (128, 128)),
            cv2.COLOR_GRAY2BGR)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.3
        fontColor = (240, 20, 20)
        lineType = 1
        y = 10
        for line in resultText:
            cv2.putText(preprocess_toshow, line,
                        (10, y),
                        font,
                        fontScale,
                        fontColor,
                        lineType)
            y += 10
        result = np.hstack((cv2.resize(img, (128, 128)), preprocess_toshow))
        results.append(result)
    print()

    concated = []
    col = []
    i = 0
    for result in results:
        if i == 0:
            col = result
        else:
            col = np.vstack((col, result))
        i += 1
        if i % 5 == 0:
            if len(concated) == 0:
                concated = col
            else:
                concated = np.hstack((concated, col))
            col = []
            i = 0

    cv2.imshow('result', concated)
    cv2.waitKey(0)

    # result = result.flatten()
    # i = 0
    # resultText = []

    # for r in result.astype('str'):
    #     resultText.append(action_dictionary[i]+":"+r)
    #     i += 1
    # resultText.append("prediction: "+action_dictionary[result.argmax(axis=0)])
    # return resultText
