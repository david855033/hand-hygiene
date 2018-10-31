from argparse import ArgumentParser
from extract.extract_folder import extract_folder

parser = ArgumentParser()
parser.add_argument("-s", help="path of source folder that contains videos",
                    default=r".\data\videosrc",
                    dest="videosrc_folder_path")
parser.add_argument("-d", help="path of destiny folder to put extracted frame",
                    default=r".\data\extract",
                    dest="extract_frame_folder_path")
parser.add_argument(
    "-m", help="limit the maximum of images to be extracted in one video",
    default=-1, dest="max_img_number", type=int)
parser.add_argument(
    "-r", help="ratio of skipped frame to extracted frame",
    default=3, dest="extract_ratio", type=int)

args = parser.parse_args()

extract_folder(videosrc_folder_path=args.videosrc_folder_path,
               extract_frame_folder_path=args.extract_frame_folder_path,
               max_img_number=args.max_img_number,
               extract_ratio=args.extract_ratio)
