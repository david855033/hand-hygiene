from argparse import ArgumentParser
from predict.predict import predict_video


parser = ArgumentParser()
parser.add_argument("-s", help="folder contain source to be predicted",
                    default=r".\data\videosrc\15_hook.MOV", dest="video_path")
parser.add_argument("-m", help="model path",
                    default=r".\data\model\model.h5", dest="model_path")

args = parser.parse_args()
predict_video(model_path=args.model_path, video_path=args.video_path, flip=-2)
# flip: 0 = vertical, 1=horizontal, -1 = ver&horizontal, -2=none
