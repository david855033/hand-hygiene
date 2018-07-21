import os
from share.global_setting import ACTIONS


def checkfolder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def check_action_folder(base_folder_path):
    for action in ACTIONS:
        checkfolder(os.path.join(base_folder_path, action))
