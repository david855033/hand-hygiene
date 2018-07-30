import cv2
from os import listdir
from os.path import isfile, join, splitext, basename
from share.global_setting import VIDEO_EXTENSION, ACTIONS
from share.file_util import check_action_folder


def extract_folder(videosrc_folder_path,
                   extract_frame_folder_path,
                   max_img_number, extract_ratio):
    print('\n[Extract video from folder]')
    print(' >> source: {0}'.format(videosrc_folder_path))
    print(' >> destiny: {0}'.format(extract_frame_folder_path))
    print(' >> extract ratio: {0}'.format(extract_ratio))

    check_action_folder(extract_frame_folder_path)

    for filepath in (f for f in listdir(videosrc_folder_path)):
        if not isfile(join(videosrc_folder_path, filepath)):  # 判斷是資料夾還是檔案
            continue

        file_name, file_extension = splitext(filepath)  # 判斷副檔名是否為接受的影像格式
        if file_extension.lower() not in VIDEO_EXTENSION:
            continue

        # 判斷是哪個action
        category = 'unidentified'
        for action in ACTIONS:
            if(action.lower() in file_name.lower()):
                category = action.lower()
                break
        dest_folder = join(extract_frame_folder_path, category)
        video_path = join(videosrc_folder_path, filepath)
        print(" >> extract {0} to {1}".format(filepath, dest_folder))
        extract(video_path, dest_folder, max_img_number, extract_ratio)


def extract(video_path, dest_folder, max_img_number, extract_ratio):
    cap = cv2.VideoCapture(video_path)
    video_filename, extension = splitext(basename(video_path))
    i = 1
    count = 1
    success = True
    while(success):
        success, frame = cap.read()
        if success:
            filepath = join(dest_folder, video_filename + r"%04d.jpg" % count)
            # save 1 img every (extract_ratio) frame
            if extract_ratio == 0 or i % extract_ratio == 0:
                print(' >> -- processing: {0}'.format(
                    filepath), end="\r", flush=True)

                cv2.imwrite(filepath, frame)
                count += 1
            i += 1
        if i > max_img_number and max_img_number > 0:  # 如果有設定最大張數才會生效
            break
    cap.release()
    print('')
