#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf
import pathlib
import random
import glob
import datetime
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
from sklearn.metrics import classification_report

print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)

tf.test.is_gpu_available()

### Generate Results
classification_report_out = "./Results/7_VGG19_Feature.txt"
f = open(classification_report_out, 'w')


#数据的目录
data_dir_train='./7/Feature/train'
#构造pathlib路径对象
data_root_train=pathlib.Path(data_dir_train)



# In[ ]:


#对目录进行迭代
for item in data_root_train.iterdir():
    print (item)


# In[ ]:


#提取出train所有的路径
all_image_path_train=list(data_root_train.glob('*/*'))

image_count_train=len(all_image_path_train)



# In[ ]:


#列表推导式 变为纯正路径
all_image_path_train=[str(path) for path in all_image_path_train]



# In[ ]:


#乱序
random.shuffle(all_image_path_train)



# In[ ]:


#编码 提取label 类名
label_names_train=sorted (item.name for item in data_root_train.glob('*/'))



# In[ ]:


#label 转换成编码index
label_to_index_train =dict((name,index) for index,name in enumerate(label_names_train))



# In[ ]:


#通过index得到所有图片的label
all_image_label_train=[label_to_index_train[pathlib.Path(p).parent.name] for p in all_image_path_train]



# In[ ]:


import IPython.display as display
index_to_label_train=dict((v,k) for k,v in label_to_index_train.items())


# In[ ]:


img_path_train=all_image_path_train[0]



# In[ ]:


#把前三步写到一起的函数
def load_preprocess_image_train(img_path_train):
    img_raw_train=tf.io.read_file(img_path_train)
    img_tensor_train=tf.image.decode_jpeg(img_raw_train,channels=3)
    img_tensor_train=tf.image.resize(img_tensor_train,[256,256])
    
    #数据增广
    img_tensor_train=tf.image.random_flip_left_right(img_tensor_train)
    img_tensor_train=tf.image.random_flip_up_down(img_tensor_train)
    
    #转换数据类型
    img_tensor_train=tf.cast(img_tensor_train,tf.float32)
    #标准化
    img_train=img_tensor_train/255
    return img_train


# In[ ]:


#构造train数据的tf.data
path_ds_train=tf.data.Dataset.from_tensor_slices(all_image_path_train)
image_dataset_train=path_ds_train.map(load_preprocess_image_train)
label_dataset_train=tf.data.Dataset.from_tensor_slices(all_image_label_train)


# In[ ]:


#合并image_dataset 和label_dataset
dataset_train=tf.data.Dataset.zip((image_dataset_train,label_dataset_train))


# In[ ]:





# In[ ]:


#val通道设置
data_dir_val='./7/Feature/vali'
#构造pathlib路径对象
data_root_val=pathlib.Path(data_dir_val)
data_root_val


# In[ ]:


for item in data_root_val.iterdir():
    print (item)


# In[ ]:


all_image_path_val=list(data_root_val.glob('*/*'))

image_count_val=len(all_image_path_val)



# In[ ]:


all_image_path_val=[str(path) for path in all_image_path_val]



# In[ ]:


random.shuffle(all_image_path_val)



# In[ ]:


label_names_val=sorted (item.name for item in data_root_val.glob('*/'))


# In[ ]:


label_to_index_val =dict((name,index) for index,name in enumerate(label_names_val))


# In[ ]:


all_image_label_val=[label_to_index_val[pathlib.Path(p).parent.name] for p in all_image_path_val]


# In[ ]:


index_to_label_val=dict((v,k) for k,v in label_to_index_val.items())



# In[ ]:


img_path_val=all_image_path_val[0]


# In[ ]:


#把前三步写到一起的函数
def load_preprocess_image_val(img_path_val):
    img_raw_val=tf.io.read_file(img_path_val)
    img_tensor_val=tf.image.decode_jpeg(img_raw_val,channels=3)
    img_tensor_val=tf.image.resize(img_tensor_val,[256,256])
    #转换数据类型
    img_tensor_val=tf.cast(img_tensor_val,tf.float32)
    #标准化
    img_val=img_tensor_val/255
    return img_val


# In[ ]:


#构造val数据的tf.data
path_ds_val=tf.data.Dataset.from_tensor_slices(all_image_path_val)
image_dataset_val=path_ds_val.map(load_preprocess_image_val)
label_dataset_val=tf.data.Dataset.from_tensor_slices(all_image_label_val)


# In[ ]:


#合并image_dataset 和label_dataset
dataset_val=tf.data.Dataset.zip((image_dataset_val,label_dataset_val))


# In[ ]:





# In[ ]:


#test通道设置
data_dir_test='./7/Feature/test'
data_root_test=pathlib.Path(data_dir_test)
for item in data_root_test.iterdir():
    print (item)
    
all_image_path_test=list(data_root_test.glob('*/*'))
len(all_image_path_test)
image_count_test=len(all_image_path_test)
image_count_test

all_image_path_test=[str(path) for path in all_image_path_test]

random.shuffle(all_image_path_test)

label_names_test=sorted (item.name for item in data_root_test.glob('*/'))

label_to_index_test =dict((name,index) for index,name in enumerate(label_names_test))

all_image_label_test=[label_to_index_test[pathlib.Path(p).parent.name] for p in all_image_path_test]


# In[ ]:


index_to_label_test=dict((v,k) for k,v in label_to_index_test.items())
index_to_label_test

img_path_test=all_image_path_test[0]
img_path_test


# In[ ]:


#把前三步写到一起的函数
def load_preprocess_image_test(img_path_test):
    img_raw_test=tf.io.read_file(img_path_test)
    img_tensor_test=tf.image.decode_jpeg(img_raw_test,channels=3)
    img_tensor_test=tf.image.resize(img_tensor_test,[256,256])
    #转换数据类型
    img_tensor_test=tf.cast(img_tensor_test,tf.float32)
    #标准化
    img_test=img_tensor_test/255
    return img_test


# In[ ]:


#构造test数据的tf.data
path_ds_test=tf.data.Dataset.from_tensor_slices(all_image_path_test)
image_dataset_test=path_ds_test.map(load_preprocess_image_test)
label_dataset_test=tf.data.Dataset.from_tensor_slices(all_image_label_test)


# In[ ]:


#合并image_dataset 和label_dataset
dataset_test=tf.data.Dataset.zip((image_dataset_test,label_dataset_test))


# In[ ]:


#train val test数据集创建完毕


# In[ ]:





# In[ ]:


train_dataset=dataset_train
val_dataset=dataset_val
test_dataset=dataset_test


# In[ ]:


train_count=int(image_count_train)
val_count=int(image_count_val)
test_count=int(image_count_test)
print(train_count,val_count,test_count)


# In[ ]:


BATCH_SIZE=32


# In[ ]:


train_dataset=train_dataset.shuffle(buffer_size=train_count).repeat().batch(BATCH_SIZE)
val_dataset=val_dataset.batch(BATCH_SIZE)
test_dataset=test_dataset.batch(BATCH_SIZE)

#到此为止训练数据和测试数据搭建完成啦！！！


# In[ ]:


keras=tf.keras
layers=tf.keras.layers
conv_base=keras.applications.VGG19(weights='imagenet',include_top=False)
conv_base.summary()


# In[ ]:


#之后需要展平这个网络
model=keras.Sequential()
model.add(conv_base)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(7,activation='softmax'))

#希望网络里面的权重不要动 把卷积积设置为不可训练
conv_base.trainable=False


# In[ ]:


model.summary()


# In[ ]:


model.compile(#optimizer='adam',
                        optimizer=keras.optimizers.Adam(lr=0.001),
                        loss='sparse_categorical_crossentropy',
                        metrics=['acc']                   
)
steps_per_epoch=train_count//BATCH_SIZE
validation_steps=val_count//BATCH_SIZE


# In[ ]:


#提早结束的回调函数 监听val_acc
early_stopping = EarlyStopping(
    monitor='val_acc',
    min_delta=0.0001,
    patience=30
)


# In[ ]:





# In[ ]:


start_time = time.time()

history=model.fit(train_dataset,
                            epochs=30,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=val_dataset,
                            validation_steps=validation_steps,
                            callbacks=[early_stopping])


end_time = time.time()
mins = (end_time - start_time) // 60
secs = (end_time - start_time) % 60
print("VGG19 Training Time: {}:{:.2f}".format(mins, secs))

f.write("VGG19 Training Time: {}:{:.2f}".format(mins, secs)+"\n")

# plot the training loss and accuracy
def plot_learning_curves(history):
	# plot the training loss and accuracy
    plt.figure(figsize=(8,5))
    plt.title('Training / Validation Loss and Accuracy on SegNet (CS-based photographs)',y=1.05)
    plt.grid(True)
    plt.gca().set_ylim(0,2)
    plt.gca().set_xlim(0,30)
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    #plt.xticks(np.linspace(0, 30, 7))
    plt.plot(history.history["loss"], label="training_loss")
    plt.plot(history.history["val_loss"], label="validation_loss")
    plt.plot(history.history["acc"], label="training_accuracy")
    plt.plot(history.history["val_acc"], label="validation_accuracy")
    plt.legend(loc="upper right",fontsize=10)
    plt.savefig('./Results/VGG19_Feature_7_curves.png')
    #plt.show()
	
plot_learning_curves(history)

# In[ ]:


start_time1=time.time()
#一定要加上batch那一步 而且只能一次 loss acc
model.evaluate(test_dataset,verbose=0)

from sklearn.metrics import confusion_matrix
x, y_true = [], []
i = 0
for element in test_dataset:
    i += 1
    _x, _y = element	
    x.append(_x.numpy())
    y_true.append(_y.numpy())
    if i==test_count//BATCH_SIZE:
        break

x = np.concatenate(x, axis=0)
y_true = np.concatenate(y_true)
y_pred = model.predict(x, verbose=0)
y_pred = np.argmax(y_pred, axis=-1)

confmatrix = confusion_matrix(y_true, y_pred)

end_time1 = time.time()
mins = (end_time1 - start_time1) // 60
secs = (end_time1 - start_time1) % 60
print("VGG19 Testing Time: {}:{:.2f}".format(mins, secs),(model.evaluate(test_dataset,verbose=0)))

f.write("VGG19 Testing Time: {}:{:.2f}".format(mins, secs)+"\n")

### 1. classification_report
target_names = ["1","2","4","7","8","10","11"]
print(classification_report(y_true, y_pred, target_names=target_names))

f.write(classification_report(y_true, y_pred, target_names=target_names))
f.close()

### 2. ConfusionMatrixPlot
pd.DataFrame(confmatrix).to_csv('./Results/7_VGG19_Feature_confusion_matrix.csv')

def ConfusionMatrixPlot(confmatrix_Input):    
    #pd.DataFrame(confmatrix_Input).to_csv('confusion_matrix.ffvfdvfdv')
    clsnames = np.arange(0, 7)
    tick_marks = np.arange(len(clsnames))
    plt.figure(figsize=(10, 10))
    plt.title('Confusion matrix of VGG19 for CS-based photographs',fontsize=15,pad=10)
    iters = np.reshape([[[i, j] for j in range(len(clsnames))] for i in range(len(clsnames))], (confmatrix_Input.size, 2))
    for i, j in iters:
        plt.text(j, i, format(confmatrix_Input[i, j]), fontsize=15, va='center', ha='center')  # 显示对应的数字

    plt.gca().set_xticks(tick_marks + 0.5, minor=True)
    plt.gca().set_yticks(tick_marks + 0.5, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')

    plt.imshow(confmatrix_Input, interpolation='nearest', cmap='cool')  # 按照像素显示出矩阵
    plt.xticks(tick_marks,['1','2','4','7','8','10','11']) # 7分类 横纵坐标类别名
    plt.yticks(tick_marks,['1','2','4','7','8','10','11'])
   
    plt.ylabel('Actual Species',labelpad=-5,fontsize=15)
    plt.xlabel('Predicted Species',labelpad=10,fontsize=15)
    plt.ylim([len(clsnames) - 0.5, -0.5])
    plt.tight_layout()

    cb=plt. colorbar()# heatmap
   #cb.set_label('Numbers of Predict',fontsize = 15)
    plt.savefig('./Results/VGG19_Feature_7.png')
    
matrixInput = np.array(confmatrix)

PercentageInput = (matrixInput.T / matrixInput.astype(np.float).sum(axis=1)).T

#print (PercentageInput)

AroundPercentageInput = np.around(PercentageInput, decimals=3)

print (AroundPercentageInput)

ConfusionMatrixPlot(AroundPercentageInput)