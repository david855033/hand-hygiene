from argparse import ArgumentParser
from cnn_module import train


parser = ArgumentParser()
parser.add_argument("-s", help="folder that contains source img",
                    default=r".\preprocess", dest="source_path")

source_path = parser.parse_args().source_path

train(source_path=source_path)