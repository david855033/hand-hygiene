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
        ground_truth = 'unidentified'
        for action in ACTIONS:
            if(action.lower() in path.lower()):
                ground_truth = action.lower()
                break

        img = cv2.imread(path)
        preprocessed_img = preprocess_img(img)

        data = np.reshape(preprocessed_img, (1, 128, 128, 1))
        predict_result = model.predict(data)

        resultText, prediction = parse_predict(predict_result, ground_truth)

        preprocess_toshow = cv2.cvtColor(
            cv2.resize(preprocessed_img, (128, 128)),
            cv2.COLOR_GRAY2BGR)

        putText(preprocess_toshow, resultText,  prediction, ground_truth)

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


def parse_predict(predict_result, ground_truth=""):
    predict_result = predict_result.flatten()
    i = 0
    resultText = []
    prediction = ACTIONS[predict_result.argmax(axis=0)]
    resultText.append("prediction: "+prediction)
    if ground_truth != "":
        resultText.append('ground_truth:'+ground_truth)
    for r in predict_result.astype('str'):
        resultText.append(ACTIONS[i]+":"+r)
        i += 1
    return resultText, prediction


def putText(preprocess_toshow, resultText, prediction, ground_truth="",
            fontScale=0.3, lineType=1, dy=10):
    if prediction == ground_truth or ground_truth == "":
        fontColor = (240, 240, 20)
    else:
        fontColor = (20, 20, 240)
    font = cv2.FONT_HERSHEY_SIMPLEX

    y = dy
    for line in resultText:
        cv2.putText(preprocess_toshow, line,
                    (10, y),
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        y += dy


def predict_video(model_path, video_path):
    print('\n[Predict frames while playing video]')
    print(' >> model_path: {0}'.format(model_path))
    print(' >> video_path: {0}'.format(video_path))

    model = load_model(model_path)
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    spf = int(1000/fps)
    print('fps: {0}'.format(fps))
    while(True):
        # Capture frame-by-frame
        ret, frame = capture.read()

        # Our operations on the frame come here
        frame_resize = cv2.resize(frame, (128, 128))
        preprocess_frame = preprocess_img(frame_resize)

        data = np.reshape(preprocess_frame, (1, 128, 128, 1))
        predict_result = model.predict(data)
        resultText, prediction = parse_predict(predict_result)

        frame_toshow = cv2.resize(frame, (256, 256))
        putText(frame_toshow, resultText,  prediction,
                fontScale=0.5, lineType=1, dy=15)

        # Display the resulting frame
        cv2.imshow('video', frame_toshow)
        if cv2.waitKey(spf) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    capture.release()
    cv2.destroyAllWindows()
