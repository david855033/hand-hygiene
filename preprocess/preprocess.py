import cv2
from os import listdir
from os.path import isfile, join, splitext, basename
from share.file_util import check_action_folder
from share.global_setting import ACTIONS


def preprocess_folder(source_path, preprocess_path):
    print('\n[Preprocess from folder]')
    print(' >> source: {0}'.format(source_path))
    print(' >> destiny: {0}'.format(preprocess_path))

    check_action_folder(preprocess_path)
    count = 0
    for action in ACTIONS:
        source_path_action = join(source_path, action)
        dest_path_action = join(preprocess_path, action)

        for filepath in (f for f in listdir(source_path_action)):

            filename, file_extention = splitext(filepath)
            if file_extention.lower() != ".jpg":
                continue

            img_source_path = join(source_path_action, filepath)
            img_dest_path = join(dest_path_action, filepath)

            img = cv2.imread(img_source_path)
            img = preprocess_img(img)
            cv2.imwrite(img_dest_path, img)
            print(
                " >> preprocessing: {0}->{1}".format(
                    filepath, dest_path_action) + " " * 20,
                end="\r")
            count += 1
    print("\n{0} imgs were preprocessed.".format(count))


def preprocess_img(image):
    """
    crop and resize the image
    without changing pixel encoding(origin: 0-255)
    """
    height, width, channel = image.shape
    if height/width < 480/640:
        estimate_width = width/640*480
        crop_width = round((width-estimate_width)/2)
        image = image[:, crop_width: crop_width + round(estimate_width), :]
    image = cv2.resize(image, (224, 224))
    return image
