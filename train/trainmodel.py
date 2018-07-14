from os.path import exists, join
from share.global_setting import ACTIONS


def trainmodel(source_path, model_save_path):
    print('\n[train model]')
    print(' >> traning data src: {0}'.format(source_path))
    print(' >> model destination: {0}'.format(model_save_path))

    if exists(model_save_path):
        input('"{0}"'
              .format(model_save_path) +
              " exsists, press Enter to overwrite. " +
              "Press Ctrl+C > Enter to Abort.")

    x_train, y_train = loadImgs(source_path)


def loadImgs(source_path):
    x_train, y_train = [], []
    for action in ACTIONS:
        action_folder_path = join(source_path, action)
        print(" >> Loading training data from: "+action_folder_path, end="")
        print("")
    return x_train, y_train
