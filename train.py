from argparse import ArgumentParser
from train.trainmodel import trainmodel

parser = ArgumentParser()
parser.add_argument("-s",
                    help="folder that contains preprocessed img for training",
                    default=r".\data\preprocess", dest="source_path")
parser.add_argument("-m", help="path of model to save",
                    default=r"",
                    dest="model_save_path")
parser.add_argument("-a", help="path of model to autosave",
                    default=r"",
                    dest="model_autosave_path")

arg = parser.parse_args()

trainmodel(source_path=arg.source_path,
           model_save_path=arg.model_save_path,
           model_autosave_path=arg.model_autosave_path)
