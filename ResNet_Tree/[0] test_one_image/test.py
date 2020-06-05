import cv2
import os
import warnings
import numpy as np
from glob import glob
from time import time
from argparse import ArgumentParser

import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Reshape, LeakyReLU
from keras.layers.core import Dense, Flatten, Dropout, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D, AveragePooling2D, GlobalMaxPooling2D
from keras import layers
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file
from keras.utils import layer_utils

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD, RMSprop

def verification(model,image):
    
    nb_classes = ["01", "02", "03", "04", "05",
                "06", "07", "08", "09", "10",
                "11", "12", "13", "14"]
    img = cv2.imread(image)
    img = img[:, :, [2, 1, 0]]
    img = img / 255
    img = np.expand_dims(img, axis=0)

    index = model.predict(img)
    list_predict = np.ndarray.tolist(index[0])
    
    max_index = list_predict.index(max(list_predict))
    

    print("{}".format(nb_classes[max_index]))
    




def addParser():
    
    args = ArgumentParser()
    args.add_argument("-image", dest='image')
    args.add_argument("-weight", dest="weight")

    return args.parse_args()

def ExecuteTime(start_time, end_time):
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    mins = int((total_time - (hours * 60)) // 60) 
    secs = int(total_time - (hours * 3600) - (mins * 60))

    return hours , mins, secs


if __name__ == "__main__":
    args = addParser()

    if args.weight is not None:
        start_time = time()
        model = load_model(args.weight)
        verification(model=model, image=args.image)
        end_time = time()
        h, m, s = ExecuteTime(start_time, end_time)
        print("Execute: {}:{}:{}".format(h, m, s))
    else:
        print("Input Error")