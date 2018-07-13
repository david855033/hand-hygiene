import cv2


def preprocess(image):
    img = cv2.resize(image, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
