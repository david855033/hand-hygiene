from flask import Flask, render_template, Response
from flask_socketio import SocketIO, send, emit
import cv2
import numpy as np
import threading
import base64
from video import VideoGet, VideoShow
from keras.models import load_model

import os
import sys
import inspect
current_dir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from preprocess.preprocess import preprocess_img
from share.global_setting import ACTIONS
from predict.predict import parse_predict

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('message')
def handle_message(message):
    print('received message: ' + message)


camera = VideoGet(2)
camera.start()

# video_shower = VideoShow(camera.frame).start()

DO_PREDICT = False


def job():
    model = object()
    if DO_PREDICT:
        model = load_model('.\data\model\model.h5')
    while True:
        try:
            frame = camera.frame

            preprocess_frame = preprocess_img(frame)
            toshow = preprocess_frame
            preprocess_frame = cv2.cvtColor(preprocess_frame,
                                            cv2.COLOR_BGR2RGB)
            preprocess_frame = np.array(preprocess_frame).astype('float32')
            preprocess_frame /= 255
            data = np.reshape(preprocess_frame, (1,)+preprocess_frame.shape)

            prediction = ""
            if DO_PREDICT:
                predict_result = model.predict(data)
                resultText, prediction = parse_predict(predict_result)

            success, img_encoded = cv2.imencode('.jpg', toshow)

            with app.app_context():
                socketio.emit('image', {'image': base64.b64encode(
                    img_encoded).decode(), 'prediction': prediction})

            socketio.sleep(0.1)  # control frame rate
        except Exception as e:
            print(e)
            socketio.sleep(0.1)


t = threading.Thread(target=job)
t.setDaemon(True)
t.start()


if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=5000)
