from argparse import ArgumentParser
from predict_module import predict


parser = ArgumentParser()
parser.add_argument("-i", help="image source to predict", dest="img_source")
parser.add_argument("-m", help="using model",
                    default=r"./models/model.h5", dest="model")

predict(model, img_source)
