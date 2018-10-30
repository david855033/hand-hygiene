import cv2
import os
import numpy as np
from os.path import join, split
from keras.models import load_model
from share.global_setting import ACTIONS
from preprocess.preprocess import preprocess_img
from random import shuffle


def predict_folder(model_path, img_folder, assign_set=[]):
    print('\n[Predict imgs in a folder]')
    print(' >> model_path: {0}'.format(model_path))
    print(' >> img_folder: {0}'.format(img_folder))

    model = load_model(model_path)
    img_list = []
    for root, dirs, files in os.walk(img_folder):
        for file in files:
            path = join(root, file)
            if len(assign_set) == 0 or int(file.split("_")[0]) in assign_set:
                img_list.append(path)
    while True:
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
            preprocessed_img = cv2.cvtColor(preprocessed_img,
                                            cv2.COLOR_BGR2RGB)
            preprocessed_img = np.array(preprocessed_img).astype('float32')
            preprocessed_img /= 255

            data = np.reshape(preprocessed_img, (1,) + preprocessed_img.shape)
            predict_result = model.predict(data)
            resultText, prediction = parse_predict(predict_result)

            img_toshow = cv2.resize(img, (128, 128))

            result_area = np.zeros((128, 128, 3), np.uint8)

            putText(result_area, resultText, prediction, ground_truth)

            result = np.hstack((img_toshow, result_area))

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
        if cv2.waitKey(5000) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def parse_predict(predict_result, ground_truth=""):
    predict_result = predict_result.flatten()
    i = 0
    resultText = []
    prediction = ACTIONS[predict_result.argmax(axis=0)]
    resultText.append("prediction: "+prediction)
    if ground_truth != "":
        resultText.append('ground_truth:'+ground_truth)
    for r in predict_result:
        resultText.append(ACTIONS[i]+":" + "%.3f" % r)
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


def predict_video(model_path, video_path, flip=-2,  DO_PREDICT=True, average_n=3, threshold=0.7):
    print('\n[Predict frames while playing video]')
    print(' >> model_path: {0}'.format(model_path))
    print(' >> video_path: {0}'.format(video_path))

    model = object()
    if DO_PREDICT:
        model = load_model(model_path)
    # r'.\data\videosrc\4_finger.MOV'
    capture = cv2.VideoCapture(0)

    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = 5  # overwrite fps
    spf = int(1000/fps)
    print('fps: {0}'.format(fps))
    average_predict_result = np.array([0, 0, 0, 0, 0, 0, 0])
    predict_result = np.array([1, 0, 0, 0, 0, 0, 0])

    while(True):
        # Capture frame-by-frame
        ret, frame = capture.read()

        # Our operations on the frame come here
        preprocess_frame = preprocess_img(frame)
        if flip != -2:
            preprocess_frame = cv2.flip(preprocess_frame, flip)

        preprocess_frame_RGB = cv2.cvtColor(preprocess_frame,
                                            cv2.COLOR_BGR2RGB)
        preprocess_frame_RGB = np.array(preprocess_frame_RGB).astype('float32')
        preprocess_frame_RGB /= 255

        data = np.reshape(preprocess_frame_RGB, (1,)+preprocess_frame_RGB.shape)

        if DO_PREDICT:
            predict_result = model.predict(data)

        frame_toshow = cv2.resize(frame, (640, 360))

        average_predict_result = (
            average_predict_result*(average_n-1)+predict_result)/average_n

        result_block = getResultBlock(predict_result, average_predict_result)
        display = np.vstack((frame_toshow, result_block))
        # Display the resulting frame
        cv2.imshow('video', display)
        if cv2.waitKey(spf) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    capture.release()
    cv2.destroyAllWindows()


def getResultBlock(predict_result, average_predict_result):
    col1 = np.vstack(getBars(predict_result))
    col2 = np.vstack(getBars(average_predict_result))
    padding = getBox(200, 5, (40, 40, 40))
    right = np.vstack(
        (getSequence(70, 470), np.zeros((130, 470, 3), dtype=np.uint8)))
    return np.hstack((col1, padding, col2, padding, right))


def getSequence(height, width,   padding=3):
    mainbox = np.zeros((height, width, 3), dtype=np.uint8)
    n = len(ACTIONS)
    boxwidth = round((width - padding * (n+1)) / n)

    for i in range(n):
        left = padding + (padding+boxwidth) * i
        mainbox[padding:-padding,
                left:left+boxwidth, 0] = 255
    return mainbox


def getBars(predict_result):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontColor = (255, 255, 255)
    fontScale = 0.3
    lineType = 1
    boxes = []
    colormap = [((255, 48, 48), (58, 13, 13)),
                ((237, 237, 42), (58, 52, 13)),
                ((94, 237, 42), (24, 58, 13)),
                ((42, 237, 149), (13, 58, 37)),
                ((42, 162, 237), (13, 42, 58)),
                ((75, 42, 237), (27, 13, 58)),
                ((224, 42, 237), (58, 13, 58))]
    for i, e in enumerate(predict_result[0]):
        e = e.item()
        w = round(e*80)
        box = np.vstack((
            np.hstack((
                getBox(24, w, colormap[i][0]),
                getBox(24, 80-w, colormap[i][1])
            )),
            getBox(1, 80, (0, 0, 0))
        ))
        cv2.putText(box, ACTIONS[i]+":"+"%.2f" % e, (5, 15),
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        boxes.append(box)
    boxes.append(getBox(25, 80, (40, 40, 40)))
    return boxes


def getBox(height, width, color=(0, 0, 0)):
    box = np.zeros((height, width, 3), dtype=np.uint8)
    box[:, :, 0] = color[0]
    box[:, :, 1] = color[1]
    box[:, :, 2] = color[2]
    return box
