from argparse import ArgumentParser
from predict.predict import predict_folder


parser = ArgumentParser()
parser.add_argument("-s", help="folder contain source to be predicted",
                    default=r".\data\extract", dest="img_folder")
parser.add_argument("-m", help="model path",
                    default=r".\data\model\model.h5", dest="model_path")

args = parser.parse_args()
predict_folder(model_path=args.model_path, img_folder=args.img_folder)

# img_source = parser.parse_args().img_source
# model = parser.parse_args().model

# image = loadImage(img_source)
# preprocess_image = preprocess(image)
# preprocess_image_to_model = np.reshape(preprocess_image, (1, 128, 128, 1))

# result = predict(model, preprocess_image_to_model)

# imageResize = cv2.resize(image, (256, 256))
# preprocessImageResize = cv2.cvtColor(cv2.resize(
#     preprocess_image, (256, 256)), cv2.COLOR_GRAY2BGR)

# predictresultshow = np.zeros((256, 256, 3), np.uint8)
# font = cv2.FONT_HERSHEY_SIMPLEX
# fontScale = 0.5
# fontColor = (255, 255, 255)
# lineType = 2
# y = 20
# for line in result:
#     cv2.putText(predictresultshow, line,
#                 (20, y),
#                 font,
#                 fontScale,
#                 fontColor,
#                 lineType)
#     y += 24
# numpy_horizontal = np.hstack(
#     (imageResize, preprocessImageResize, predictresultshow))

# cv2.imshow("prediction", numpy_horizontal)
# cv2.waitKey(0)
