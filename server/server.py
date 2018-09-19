from flask import Flask, render_template, Response
from flask_socketio import SocketIO, send, emit
import cv2
import threading
import base64
from video import VideoGet, VideoShow

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)


@app.route('/')  # 主頁
def index():
        # jinja2模板，具體格式保存在yemplates/index.html文檔中
    return render_template('index.html')


@socketio.on('message')
def handle_message(message):
    print('received message: ' + message)


camera = VideoGet(1)
camera.start()

# video_shower = VideoShow(camera.frame).start()


def job():
    while True:
        try:
            frame = camera.frame
            # video_shower.frame = frame
            success, img_encoded = cv2.imencode('.jpg', frame)
            with app.app_context():
                socketio.emit('image', base64.b64encode(img_encoded).decode())
            socketio.sleep(0.1)
        except Exception as e:
            print(e)
            socketio.sleep(2)

    # while True:
    #     ret, frame = webcam.read()
    #     _, img_encoded = cv2.imencode('.jpg', frame)
    #     # with app.app_context():
    #     #     socketio.emit('image', base64.b64encode(img_encoded).decode())
    #     cv2.imshow('window', frame)
    #     socketio.sleep(0.5)

    # webcam.release()
    # cv2.destroyAllWindows()


t = threading.Thread(target=job)
t.setDaemon(True)
t.start()


if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=5000)
