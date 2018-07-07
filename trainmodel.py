from argparse import ArgumentParser
from cnn_module import train

parser = ArgumentParser()
parser.add_argument("-f", help="folder that contains source img",
                    default=r".\preprocess", dest="source_path")
parser.add_argument("-s", help="name of model to save",
                    default="model", dest="model_save_name")
parser.add_argument("-sl", help="name of model to load",
                    default="model", dest="model_load_name")

source_path = parser.parse_args().source_path
model_save_name = parser.parse_args().model_save_name
model_load_name = parser.parse_args().model_load_name

train(source_path=source_path, model_load_name=model_load_name,
      model_save_name=model_save_name)
