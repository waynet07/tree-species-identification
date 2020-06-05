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
from sklearn.metrics import classification_report

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD, RMSprop

import matplotlib.pyplot as plt

import seaborn as sb
import pandas as pd

from net.vgg import VGG16, VGG19

from net.resnet import resnet50



class Train(object):

    def __init__(self, train_folder, vali_folder, weights_file, storage_file, batch_size=8, min_delta=1e-3, patience=3):
        
        self._nb_classes, self._train_genrator, self._vali_generator = self._dataGenerator(train_folder, vali_folder)
        
        self._weights_file = weights_file
        self._storage_file = storage_file
        self._batch_size = batch_size
        self._checkpoint = ModelCheckpoint(storage_file, monitor='val_acc', verbose=1, save_best_only=True)
        self._monitor = EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience, verbose=1, mode='auto', restore_best_weights=True)

    def _dataGenerator(self, train_folder, vali_folder):

        nb_class = len(os.listdir(train_folder))

        train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
        )

        vali_datagen = ImageDataGenerator(
        rescale=1./255,
        )

        train_generator = train_datagen.flow_from_directory(
        train_folder,
        target_size=(256, 256),
        batch_size=8,
        class_mode='categorical'
        )

        vail_generator = vali_datagen.flow_from_directory(
        vali_folder,
        target_size=(256, 256),
        batch_size=8,
        class_mode='categorical'
        )

        return nb_class, train_generator, vail_generator

    def vgg19(self, epochs=30):

        model = VGG19(include_top=False, input_shape=(256, 256, 3), weights_file=self._weights_file)
        last_layer = model.get_layer("pool5").output
        x = Flatten(name='flatten')(last_layer)
        x = Dense(1024, activation='relu', name='fc6')(x)
        #x = Dropout(0.2)(x)
        x = Dense(2048, activation='relu', name='fc7')(x)
        #x = Dropout(0.3)(x)
        out = Dense(self._nb_classes, activation='softmax', name='fc8')(x)
        train_model = Model(model.input, out)
        train_model.compile(loss='categorical_crossentropy',
                            optimizer=SGD(lr=1e-4, momentum=0.9),
                            metrics=['accuracy'])

        steps_per_epoch =  self._train_genrator.n // self._batch_size
        validation_steps =  self._vali_generator.n // self._batch_size
        #callbacks =[self._checkpoint, self._monitor]
        
        history = train_model.fit_generator(
            self._train_genrator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=self._vali_generator,
            validation_steps=validation_steps)
            #callbacks=callbacks)
        
        return history

    def Resnet50(self, epochs=30):
        model = resnet50(include_top=False, input_shape=(256, 256, 3))

        last_layer = model.get_layer("avg_pool").output
        x = Flatten(name='flatten')(last_layer)
        x = Dropout(0.3)(x)
        out = Dense(self._nb_classes, activation='softmax', name='fc8')(x)
        train_model = Model(model.input, out)
        train_model.compile(loss='categorical_crossentropy',
                            optimizer=SGD(lr=1e-4, momentum=0.9),
                            metrics=['accuracy'])

        steps_per_epoch =  self._train_genrator.n // self._batch_size
        validation_steps =  self._vali_generator.n // self._batch_size

        #callbacks =[self._checkpoint, self._monitor]
        
        history =  train_model.fit_generator(
            self._train_genrator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=self._vali_generator,
            validation_steps=validation_steps)
            #callbacks=callbacks)
        train_model.save(self._storage_file)
        return history
        
        
        
    def pltTrainAccuracy(self, title, history):

        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title(title)
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(title + "_accuracy" + ".jpg")
    
    def pltTrainLoss(self, title, history):
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title(title)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(title + "_loss" + ".jpg")
  
    def plot_learning_curves(self, title, name, size, history):
	# plot the training loss and accuracy
        Titile = "Training / Validation Loss and Accuracy on ResNet50 (" + name + " Photographs)"
        plt.figure(figsize=(8,5))
        plt.title(Titile,y=1.05)
        plt.grid(True)
        plt.gca().set_ylim(0,2)
        plt.gca().set_xlim(0,30)
        plt.xlabel('Epoch')
        plt.ylabel('Loss/Accuracy')
    #plt.xticks(np.linspace(0, 30, 7))
        plt.plot(history.history["loss"], label="training_loss")
        plt.plot(history.history["val_loss"], label="validation_loss")
        plt.plot(history.history["accuracy"], label="training_accuracy")
        plt.plot(history.history["val_accuracy"], label="validation_accuracy")
        plt.legend(loc="upper right",fontsize=10)
        SaveFile = "./Results/ResNet50_"+name+"_"+size+"_curves.png"
        FileName = "./Results/ResNet50_" + name + "_" + size +"_curves.png"
        plt.savefig(FileName)    


class Test(object):

    def __init__(self, test_folder, model_file):
        self._test_folder = test_folder
        self._nb_classes = os.listdir(test_folder)
        print(self._nb_classes)
        self._model = load_model(model_file)
        self._class_acc = {}
        self._confusion_matrix = np.zeros((len(self._nb_classes), len(self._nb_classes)), dtype='int32')
    
    @property
    def class_acc(self):
        return self._class_acc
    
    def verification(self):
        y_true = []
        y_pred = []
        all_acc = 0
        all_total = 0
        for p in self._nb_classes:

            images = glob(os.path.join(self._test_folder, p, "*.jpg"))
            acc = 0
            total = 0

            for image in images:

                img = cv2.imread(image)
                img = img[:, :, [2, 1, 0]]
                img = img / 255
                img = np.expand_dims(img, axis=0)

                index = self._model.predict(img)
                list_predict = np.ndarray.tolist(index[0])
                
                max_index = list_predict.index(max(list_predict))
                self._confusion_matrix[self._nb_classes.index(p)][max_index] += 1
                y_true.append(p)
                y_pred.append(self._nb_classes[max_index])
                #print("{}:{}".format(p, self._nb_classes[max_index]))
                if self._nb_classes[max_index] == p:
                    acc += 1
                    all_acc += 1
                all_total += 1
                total += 1
            print("{} verified!".format(p))
            self._class_acc[p] = round(acc/ total, 2)
		
        return y_true, y_pred

    def printResult(self):
        acc = 0
        
        for p in self._nb_classes:
            print("{}: {}".format(p, self._class_acc[p]))
            acc += self._class_acc[p]

        return (acc/len(self._nb_classes))
    
    def confusionMatrix(self, title):

        labels = self._nb_classes
        df_cm = pd.DataFrame(self._confusion_matrix, index=labels, columns=labels)
        sb.set(font_scale=1.3)
        fig = plt.figure(figsize=(10, 10))
        heat_map = sb.heatmap(df_cm, fmt='d',
                            cmap='BuPu', annot=True, cbar=True, center=True,
                            linewidths=0.5, linecolor='w', square=True,
                            cbar_kws={'label': 'Number of Prediction', 'orientation': 'vertical'})
        
        heat_map.set_xticklabels(heat_map.get_xticklabels(), rotation=0, fontsize=10)
        heat_map.set_yticklabels(heat_map.get_yticklabels(), rotation=0, fontsize=10)
        heat_map.xaxis.set_ticks_position("bottom")
        heat_map.set_ylim(len(self._nb_classes), 0)
        plt.title("Confusion matrix : {}".format(title))
        plt.xlabel("Predicted Tree", labelpad=10)
        plt.ylabel("True Tree", labelpad=10)
        plt.savefig("confusion_" +  title + ".jpg")

    def ConfusionMatrixPlot_Paper(self, title, size, name):
        print (name)
        print (size)
        PercentageInput = (self._confusion_matrix.T / self._confusion_matrix.astype(np.float).sum(axis=1)).T
        AroundPercentageInput = np.around(PercentageInput, decimals=3)
        #print (PercentageInput)
        print (AroundPercentageInput)
	
        if size == "7":
            # pd.DataFrame(confmatrix).to_csv('./Results/7_VGG19_Original_confusion_matrix.csv')
            clsnames = np.arange(0, 7)
            tick_marks = np.arange(len(clsnames))
            plt.figure(figsize=(10, 10))
            Title = "Confusion matrix of ResNet for " + name + " photographs"
            plt.title(Title,fontsize=15,pad=10)
            iters = np.reshape([[[i, j] for j in range(len(clsnames))] for i in range(len(clsnames))], (AroundPercentageInput.size, 2))
            for i, j in iters:
                plt.text(j, i, format(AroundPercentageInput[i, j]), fontsize=15, va='center', ha='center')  # 显示对应的数字
            plt.gca().set_xticks(tick_marks + 0.5, minor=True)
            plt.gca().set_yticks(tick_marks + 0.5, minor=True)
            plt.gca().xaxis.set_ticks_position('none')
            plt.gca().yaxis.set_ticks_position('none')
            plt.grid(True, which='minor', linestyle='-')

            plt.imshow(AroundPercentageInput, interpolation='nearest', cmap='cool')  # 按照像素显示出矩阵
            plt.xticks(tick_marks,['1','2','4','7','8','10','11']) # 7分类 横纵坐标类别名
            plt.yticks(tick_marks,['1','2','4','7','8','10','11'])
   
            plt.ylabel('Actual Species',labelpad=-5,fontsize=15)
            plt.xlabel('Predicted Species',labelpad=10,fontsize=15)
            plt.ylim([len(clsnames) - 0.5, -0.5])
            plt.tight_layout()
            cb=plt. colorbar()# heatmap
            plt.savefig("./Results/"+"cm_" +  size  + "_" + name + ".jpg")
        elif size == "14":
            clsnames = np.arange(0, 14)
            tick_marks = np.arange(len(clsnames))
            plt.figure(figsize=(10, 10))
            Title = "Confusion matrix of ResNet for " + name + " photographs"
            plt.title(Title,fontsize=15,pad=10)
            iters = np.reshape([[[i, j] for j in range(len(clsnames))] for i in range(len(clsnames))], (AroundPercentageInput.size, 2))
            for i, j in iters:
                plt.text(j, i, format(AroundPercentageInput[i, j]), fontsize=12, va='center', ha='center')  # 熱力圖 對應文字

            plt.gca().set_xticks(tick_marks + 0.5, minor=True)
            plt.gca().set_yticks(tick_marks + 0.5, minor=True)
            plt.gca().xaxis.set_ticks_position('none')
            plt.gca().yaxis.set_ticks_position('none')
            plt.grid(True, which='minor', linestyle='-')

            plt.imshow(AroundPercentageInput, interpolation='nearest', cmap='cool')  # 按照像素顯示矩陣
            plt.xticks(tick_marks, clsnames+1)
            plt.yticks(tick_marks, clsnames+1)
            plt.ylabel('Actual Species',labelpad=-5,fontsize=15)
            plt.xlabel('Predicted Species',labelpad=10,fontsize=15)
            plt.ylim([len(clsnames) - 0.5, -0.5])
            plt.tight_layout()

            cb=plt. colorbar()# 熱力圖
   #cb.set_label('Numbers of Predict',fontsize = 15)
            plt.savefig("./Results/"+"cm_" +  size  + "_" + name + ".jpg")

def addParser():
    
    args = ArgumentParser()
    args.add_argument("-type", type=int, default=0, dest="type")
    args.add_argument("-folder", dest='folder')
    args.add_argument("-network", default='resnet', dest='net')
    args.add_argument("-weight", dest="weight")
    args.add_argument("-size", dest="size")
    args.add_argument("-name", dest="name")
	
    return args.parse_args()

def ExecuteTime(start_time, end_time):
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    mins = int((total_time - (hours * 60)) // 60) 
    secs = int(total_time - (hours * 3600) - (mins * 60))

    return hours , mins, secs


if __name__ == "__main__":

    args = addParser()

    if args.type == 0: # Train
	### Generate Results
        classification_report_out = "./Results/Training_" + args.size + "_ResNet50_" + args.name + ".txt"
        f = open(classification_report_out, 'w')

        start_time = time()
        train_folder = os.path.join("./" + args.folder, "train")
        vali_folder = os.path.join("./" + args.folder, "vali")
        train = Train(train_folder=train_folder, vali_folder=vali_folder, weights_file=None, storage_file=args.weight)
        if args.net == "resnet":
            history = train.Resnet50()
       # elif args.net == "vgg":
       #     history = train.vgg19()
        end_time = time()
        hours, mins, secs = ExecuteTime(start_time, end_time)
        print("Execute time: {}: {}:{}".format(hours, mins, secs))
        f.write("Training_time: {}: {}:{}".format(hours, mins, secs))
        f.close()
     #   train.pltTrainAccuracy(title="Accuracy " + args.weight[:-5], history=history)
     #   train.pltTrainLoss(title="Loss " + args.weight[:-5], history=history)
        train.plot_learning_curves(title="Loss " + args.weight[:-5], name=args.name, size=args.size, history=history)
    elif args.type == 1: # Test
		### Generate Results
        classification_report_out = "./Results/Testing_" + args.size + "_ResNet50_" + args.name + ".txt"
        f = open(classification_report_out, 'w')
		
        test_folder = os.path.join("./" + args.folder, "test")
        if args.weight is not None:
            start_time = time()
            test = Test(test_folder=test_folder, model_file=args.weight)
            y_true, y_pred = test.verification()
        else:
            TypeError("Weight file is None.")
        end_time = time()
        hours, mins, secs = ExecuteTime(start_time, end_time)
        #print("y_true:{}".format(y_true))
       # print("y_pred:{}".format(y_pred))
		
		### 1. classification_report
		
        print ("size"+args.size)
        if args.size == "14":
            target_names = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14"]
        elif args.size == "7":
            target_names = ["1","2","4","7","8","10","11"]
        
        f.write(classification_report(y_true, y_pred, target_names=target_names))
        f.write("Testing time: {}: {}:{}".format(hours, mins, secs))
        f.close()

        print(classification_report(y_true, y_pred, target_names=target_names))
        print("Accuracy:{}".format(test.class_acc))
        print("Execute time: {}: {}:{}".format(hours, mins, secs))
        #test.confusionMatrix("Confusion Matrix " + args.weight[:-5])
        
        print()
        print("Confusion Matrix " + args.weight[:-5])
        
        test.ConfusionMatrixPlot_Paper("Confusion Matrix " + args.weight[:-5], args.size, args.name)
		