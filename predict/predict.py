import cv2
import os
import numpy as np
import keras
from keras.utils.generic_utils import CustomObjectScope
from os.path import join, split
from keras.models import load_model
from share.global_setting import ACTIONS
from preprocess.preprocess import preprocess_img
from random import shuffle
from keras.utils.generic_utils import CustomObjectScope
import time


def predict_folder(model_path, img_folder, assign_set=[]):
    print('\n[Predict imgs in a folder]')
    print(' >> model_path: {0}'.format(model_path))
    print(' >> img_folder: {0}'.format(img_folder))
    with CustomObjectScope({'relu6': keras.layers.ReLU(6.), 'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
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


def predict_video(
        model_path, video_path, flip=-2, DO_PREDICT=True, average_n=3,
        threshold=0.4):
    print('\n[Predict frames while playing video]')
    print(' >> model_path: {0}'.format(model_path))
    print(' >> video_path: {0}'.format(video_path))

    model = object()
    if DO_PREDICT:
        with CustomObjectScope({'relu6': keras.layers.ReLU(6.),
                                'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
            model = load_model(model_path)

    capture = cv2.VideoCapture(r'.\data\testvideo\all.MOV')
    video_mode = True

    # set start position
    capture.set(cv2.CAP_PROP_POS_FRAMES, 60)

    fps = capture.get(cv2.CAP_PROP_FPS)
    # fps = 40  # overwrite fps
    spf = 1  # int(1000/fps)
    print('fps: {0}'.format(fps))

    out_video = cv2.VideoWriter(
        r'data\output\output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
        fps, (480, 622))

    average_predict_result = np.array([[0, 0, 0, 0, 0, 0, 0]])
    predict_result = np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
    action_status = np.zeros(6)
    timestamp = time.time()
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

        # preprocess_frame_RGB /= 255
        preprocess_frame_RGB = keras.applications.mobilenet.preprocess_input(
            preprocess_frame_RGB)

        data = np.reshape(preprocess_frame_RGB, (1,) +
                          preprocess_frame_RGB.shape)

        if DO_PREDICT:
            predict_result = model.predict(data)

        frame_toshow = cv2.resize(preprocess_frame, (480, 360))

        newtimestamp = time.time()
        duration = newtimestamp - timestamp
        timestamp = newtimestamp
        if video_mode:
            duration = 1/fps

        max_index = np.argmax(predict_result, axis=1)[0]
        one_hot_result = np.zeros(predict_result.shape)
        one_hot_result[0][max_index] = 1

        average_predict_result = (
            average_predict_result * (average_n - duration) + one_hot_result *
            duration) / average_n

        action_status = update_status(
            action_status, average_predict_result, threshold)

        result_block = getResultBlock(
            predict_result, average_predict_result, action_status, 400, 480)
        display = np.vstack((frame_toshow, result_block))

        # Display the resulting frame
        cv2.imshow('video', display)
        pos_msec = capture.get(cv2.CAP_PROP_POS_MSEC)
        if pos_msec > 41000:
            break
        out_video.write(display)
        if cv2.waitKey(spf) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    capture.release()
    out_video.release()
    cv2.destroyAllWindows()


def getResultBlock(
        predict_result, average_predict_result, action_status, height, width):
    col1 = np.vstack(
        getBars(predict_result, height=round(height/2), width=round(width/2)))
    col2 = np.vstack(
        getBars(
            average_predict_result, height=round(height / 2),
            width=width - round(width / 2)))
    bottom = np.hstack((col1, col2))
    top = getSequence(action_status, 70, width)
    return np.vstack((top, bottom))


def getBars(predict_result, height, width):
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
    bar_height = round(height/8)-1
    bar_width = width
    for i, e in enumerate(predict_result[0]):
        e = e.item()
        bright_width = round(e*bar_width)
        box = np.vstack((
            np.hstack((
                getBox(bar_height-1, bright_width, colormap[i][0]),
                getBox(bar_height-1, bar_width-bright_width, colormap[i][1])
            )),
            getBox(1, width, (0, 0, 0))
        ))
        cv2.putText(box, ACTIONS[i]+":"+"%.2f" % e, (5, 15),
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        boxes.append(box)
    boxes.append(getBox(bar_height, width, (40, 40, 40)))
    return boxes


def getBox(height, width, color=(0, 0, 0)):
    box = np.zeros((height, width, 3), dtype=np.uint8)
    box[:, :, 0] = color[0]
    box[:, :, 1] = color[1]
    box[:, :, 2] = color[2]
    return box


def getSequence(action_status, height, width,  padding=3):
    mainbox = np.zeros((height, width, 3), dtype=np.uint8)
    n = len(action_status)
    boxwidth = round((width - padding * (n+1)) / n)

    for i in range(n):
        left = padding + (padding+boxwidth) * i
        color = (12, 40, 21)
        if action_status[i] == 1:
            color = (39, 150, 15)

        box = getBox(height-padding*2, boxwidth, color)

        cv2.putText(box, ACTIONS[i+1],
                    (5, 40),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4,
                    color=(255, 255, 255), lineType=1)

        mainbox[padding:-padding, left: left +
                boxwidth] = box

    return mainbox


def update_status(action_status, average_predict_result, threshold):
    argmax = np.argmax(average_predict_result, axis=1)[0]
    if argmax > 0 and average_predict_result[0][argmax] > threshold:
        argmax -= 1
        action_status[argmax] = 1
    return action_status
