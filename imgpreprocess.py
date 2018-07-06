from argparse import ArgumentParser
from imgpreprocess_module import imgPreprocess
parser = ArgumentParser()
parser.add_argument("-s", help="folder that contains source img",
                    default=r".\dest", dest="source_path")
parser.add_argument("-p", help="folder to save ",
                    default=r".\preprocess", dest="preprocess_path")

source_path = parser.parse_args().source_path
preprocess_path = parser.parse_args().preprocess_path

imgPreprocess(source_path=source_path, preprocess_path=preprocess_path)
