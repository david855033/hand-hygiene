from os import listdir
from os.path import isfile, join
from extract_module import extract

VIDEO_EXTENSION = ['mov', 'mp4']  # use lower case


def extractFolder(source_path=r".\videosrc", dest_path=r".\dest",
                  max_img_number=-1, extract_ratio=0):

    for f in (f for f in listdir(source_path)
              if isfile(join(source_path, f)) and '.' in f):
        filename, file_extension = f.split('.')
        if(file_extension.lower() not in VIDEO_EXTENSION):
            continue
        videopath = join(source_path, f)
        dest_folder = join(dest_path, filename)
        print("extract {0} to {1}".format(videopath, dest_folder))
        extract(videopath, dest_folder, max_img_number, extract_ratio)
