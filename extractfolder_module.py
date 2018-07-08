from os import listdir
from os.path import isfile, join, splitext
from extract_module import extract
from action_dictionary import action_dictionary

VIDEO_EXTENSION = ['.mov', '.mp4']  # use lower case


def extractFolder(source_path=r".\videosrc", dest_path=r".\dest",
                  max_img_number=-1, extract_ratio=5):

    for f in (f for f in listdir(source_path)):
        if not isfile(join(source_path, f)):  # 判斷是資料夾還是檔案
            continue

        filename, file_extension = splitext(f)  # 判斷副檔名是否為接受的影像格式

        if file_extension.lower() not in VIDEO_EXTENSION:
            continue

        # 判斷是哪個action category
        category = 'unidentified'
        for action in action_dictionary:
            if(action.lower() in filename.lower()):
                category = action.lower()
                break
        videopath = join(source_path, f)
        dest_folder = join(dest_path, category)
        print("extract {0} to {1}".format(videopath, dest_folder))
        extract(videopath, dest_folder, max_img_number, extract_ratio)
