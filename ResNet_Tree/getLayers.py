import keras
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Conv2D, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from argparse import ArgumentParser

from net.vgg import VGG19
from net.resnet import resnet50
from net.segnet import SegNet



def get_layers(images, network='resnet', weight=None):
    
    storage_folder = "./" + network
    if not os.path.exists(storage_folder):
        os.mkdir(storage_folder)

    img = image.load_img(images, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.

    if network == "resnet":
        if weight is None:
            base_model = resnet50(include_top=False, weights="imagenet")
            end = -1
        else:
            base_model = resnet50(include_top=True, weights=weight, classes=14)
            end = -2
    elif network == 'vgg':
        if weight is None:
            base_model = VGG19(include_top=False,weights='imagenet') # 修改網路結構
            end = -1
        else:
            base_model = VGG19(include_top=True,weights=weight, classes=14)
            end = -3
    elif network == 'segnet':
        base_model = SegNet()
    
    layer_outputs = [layer.output for layer in base_model.layers[2:end]]

    layer_names = []
    for layer in base_model.layers[2:end]:
        layer_names.append(layer.name)

    print(layer_names)

    model = Model(inputs=base_model.input, outputs=layer_outputs)
    activations = model.predict(img)


    images_per_row = 16
    i = 0
    for activation, layer_name in zip(activations, layer_names):

        # This is the number of features in the feature map
        n_features = activation.shape[-1]
        
        # The feature map has shape (1, size, size, n_features)
        size = activation.shape[1]

        n_cols = n_features // images_per_row
        display_grid = np.zeros((size*n_cols, images_per_row * size))

        for col in range(n_cols):
            for row in range(images_per_row):

                channel_image = activation[0, :, :, col * images_per_row + row]

                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype("uint8")
                display_grid[col * size: (col + 1) * size,
                            row * size: (row + 1) * size] = channel_image
        
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        
        plt.title(layer_name, fontsize=20)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.savefig(os.path.join(storage_folder, str(i) + ".jpg"))# 儲存資料夾
        i += 1


def addParser():
    
    args = ArgumentParser()
    args.add_argument("-image", default="10-10_1.jpg", dest='img')
    args.add_argument("-network", default='resnet', dest='net')
    args.add_argument("-weight", dest="weight")
    return args.parse_args()

if __name__ == "__main__":

    args = addParser()

    get_layers(images=args.img, network=args.net, weight=args.weight)