from argparse import ArgumentParser
from preprocess.preprocess import preprocess_folder

parser = ArgumentParser()
parser.add_argument("-s", help="folder that contains source imgs",
                    default=r".\data\extract", dest="source_path")
parser.add_argument("-d", help="folder to save preprocessed imgs",
                    default=r".\data\preprocess", dest="preprocess_path")

args = parser.parse_args()

preprocess_folder(source_path=args.source_path,
                  preprocess_path=args.preprocess_path)
