from argparse import ArgumentParser
from extract_module import extract

parser = ArgumentParser()
parser.add_argument("-s", help="folder that contains source videos",
                    default=".\videosrc", dest="source_path")
parser.add_argument("-f", help="path of destiny folder",
                    default=".\dest", dest="dest_path")
parser.add_argument("-n", help="limit the maximum of images to be extracted in one video",
                    default=-1, dest="max_img_number", type=int)
parser.add_argument(
    "-r", help="ratio of frame to be skipped to frame to be extracted",
    default=0, dest="extract_ratio", type=int)

source_path= parser.parse_args().source_path
dest_path= parser.parse_args().dest_path
max_img_number= parser.parse_args().max_img_number
extract_ratio= parser.parse_args().extract_ratio