import cv2
import os
from os.path import join, splitext, basename


def extract(video_path, folder_path, max_img_number=-1, extract_ratio=0):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    cap = cv2.VideoCapture(video_path)
    video_filename, extension = splitext(basename(video_path))
    i = 1
    count = 1
    success = True
    while(success):
        success, frame = cap.read()
        if success:
            filepath = join(folder_path, video_filename+r"%04d.jpg" % count)
            # save 1 img every (extract_ratio) frame
            if extract_ratio == 0 or i % extract_ratio == 0:
                print('> processing: {0}'.format(
                    filepath), end="\r", flush=True)
                cv2.imwrite(filepath, frame)
                count += 1
            i += 1
        if i > max_img_number and max_img_number > 0:  # 如果有設定最大張數才會生效
            break
    cap.release()
    print('')
