from argparse import ArgumentParser
from predict.predict import predict_folder


parser = ArgumentParser()
parser.add_argument("-s", help="folder contain source to be predicted",
                    default=r".\data\extract", dest="img_folder")
parser.add_argument("-m", help="model path",
                    default=r".\data\model\model.h5", dest="model_path")

args = parser.parse_args()
predict_folder(
    model_path=args.model_path, img_folder=args.img_folder,
    assign_set=[5, 6, 1])
