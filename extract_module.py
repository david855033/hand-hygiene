import cv2
import os


def extract(video_path, folder_path, max_img_number=-1, extract_ratio=0):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    cap = cv2.VideoCapture(video_path)
    i = 1
    count = 1
    ret = True
    while(ret):
        ret, frame = cap.read()
        if ret:
            filename = folder_path+r"\%06d.jpg" % count
            if extract_ratio == 0 or i % extract_ratio == 0:
                print(filename)
                cv2.imwrite(filename, frame)
                count += 1
            i += 1
        if i > max_img_number and max_img_number > 0:
            break
    cap.release()
    print("end of extraction")
