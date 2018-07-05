from argparse import ArgumentParser
import numpy as np
import cv2
import os

parser = ArgumentParser()
parser.add_argument("video_path", help="path of video to be extracted")
parser.add_argument("-f", help="path of destiny folder",
                    default=".\dest", dest="folder_path")
parser.add_argument("-n", help="path of destiny folder",
                    default=-1, dest="MAX_IMG_NUMBER", type=int)

video_path = parser.parse_args().video_path
folder_path = parser.parse_args().folder_path
MAX_IMG_NUMBER = parser.parse_args().MAX_IMG_NUMBER

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

cap = cv2.VideoCapture(video_path)
i = 1
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        filename = folder_path+"\\%06d.jpg" % i
        print(filename)
        cv2.imwrite(filename, frame)
        i += 1
    if cv2.waitKey(1) & 0xFF == ord('q') or (i > MAX_IMG_NUMBER and MAX_IMG_NUMBER > 0):
        break

cap.release()
cv2.destroyAllWindows()
