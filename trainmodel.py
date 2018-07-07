from argparse import ArgumentParser
from cnn_module import train


parser = ArgumentParser()
parser.add_argument("-f", help="folder that contains source img",
                    default=r".\preprocess", dest="source_path")
parser.add_argument("-l", help="load from model",
                    default="model", dest="model_load_name")
parser.add_argument("-s", help="what name should the model be saved to",
                    default="model", dest="model_save_name")
source_path = parser.parse_args().source_path
model_load_name = parser.parse_args().model_load_name
model_save_name = parser.parse_args().model_save_name

train(source_path=source_path, model_load_name=model_load_name,
      model_save_name=model_save_name)
