import cv2
import numpy as np
from argparse import ArgumentParser
from imageloader_module import loadImage
from preprocess_module import preprocess
from predict_module import predict

parser = ArgumentParser()
parser.add_argument("-i", help="image source to predict", dest="img_source")
parser.add_argument("-m", help="using model",
                    default=r"./models/model.h5", dest="model")

img_source = parser.parse_args().img_source
model = parser.parse_args().model

image = loadImage(img_source)
preprocess_image = preprocess(image)
preprocess_image_to_model = np.reshape(preprocess_image, (1, 128, 128, 1))

result = predict(model, preprocess_image_to_model)

imageResize = cv2.resize(image, (256, 256))
preprocessImageResize = cv2.cvtColor(cv2.resize(
    preprocess_image, (256, 256)), cv2.COLOR_GRAY2BGR)

predictresultshow = np.zeros((256, 256, 3), np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
fontColor = (255, 255, 255)
lineType = 2
y = 20
for line in result:
    cv2.putText(predictresultshow, line,
                (20, y),
                font,
                fontScale,
                fontColor,
                lineType)
    y += 24
numpy_horizontal = np.hstack(
    (imageResize, preprocessImageResize, predictresultshow))

cv2.imshow("prediction", numpy_horizontal)
cv2.waitKey(0)
