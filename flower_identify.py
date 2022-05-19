# -*- coding: utf-8 -*-
"""
Created on Tue May 10 14:19:39 2022

@author: bianca
"""

#%% 载入常用第三方库
import tensorflow as tf
import numpy as np
import pandas as pd
import glob
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array
from flower_get_data import get_data,get_augmentation_data
from sklearn.utils import shuffle
#%%载入数据
# x_valid,y_valid,label2=get_data(routine='D:/tiqam_data/week4/102flowers/valid.txt')
# x,y,label=get_augmentation_data(routine='D:/tiqam_data/week4/102flowers/train.txt')
# x_test,y_test=get_data('D:/tiqam_data/week4/102flowers/test.txt')
data_dir = './train'  # 花卉文件夹的路径
data_train=tf.keras.preprocessing.image_dataset_from_directory(data_dir,
                                                               label_mode='categorical',
                                                               labels='inferred',
                                                               color_mode='rgb',
                                                               image_size=(64,64),
                                                               seed=1,batch_size=102)
data_dir2 = './valid'  # 花卉文件夹的路径
data_test = tf.keras.preprocessing.image_dataset_from_directory(data_dir2,
                                                               label_mode='categorical',
                                                               labels='inferred',
                                                               color_mode='rgb',
                                                               image_size=(64,64),
                                                               seed=1,batch_size=102)

#%% 构建模型(增加了随即丢弃权重比例)
tf.keras.backend.clear_session()
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(16,(5,5),input_shape=(64,64,3),activation='relu'))
#model.add(tf.keras.layers.Conv2D(32,(3,3),activation='relu'))
#model.add(tf.keras.layers.Conv2D(32,(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2,2))
#model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Conv2D(32,(5,5),activation='relu'))
#model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
#model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2,2))
#model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation='relu'))
#model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(256,activation='relu'))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(102,activation='softmax'))
model.compile(loss=tf.keras.losses.binary_crossentropy,optimizer='adam',metrics=['accuracy'])
model.summary()

model.fit(data_train,epochs=10,validation_data=(data_test),batch_size=238)
          # callbacks=tf.keras.callbacks.ModelCheckpoint('D:/model_save/flower.h5', 
          #               								monitor='val_loss', 
          #               								verbose=0, 
          #               								save_best_only=True, 
          #               								save_weights_only=False, 
          #               								mode='auto', 
          #               								period=1))

#%% vgg16模型
from tensorflow.python.keras.models import Model
model_vgg16 = tf.keras.applications.VGG16(weights='imagenet', include_top=False, 
                                          input_shape=(64, 64, 3))
for layer in model_vgg16.layers:
    layer.trainable = False
model = tf.keras.layers.Flatten(name='flatten')(model_vgg16.output)
model = tf.keras.layers.Dense(64, activation='relu')(model)
model = tf.keras.layers.BatchNormalization()(model)
#model = tf.keras.layers.Dropout(0.5)(model)
model = tf.keras.layers.Dense(32, activation='relu')(model)
model = tf.keras.layers.BatchNormalization()(model)
#model = tf.keras.layers.Dropout(0.5)(model)
model = tf.keras.layers.Dense(102, activation='softmax')(model)
model = Model(inputs=model_vgg16.input, outputs=model, name='vgg16')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(data_train,steps_per_epoch=10 , epochs=5, validation_data=data_test)
#%% resnet50
resnet50_fine_tune = tf.keras.models.Sequential()
resnet50_fine_tune.add(tf.keras.applications.ResNet50(include_top = False, # 网络结构的最后一层,resnet50有1000类,去掉最后一层
                                                   pooling = 'avg', #resnet50模型倒数第二层的输出是三维矩阵-卷积层的输出,做pooling或展平
                                                   weights = 'imagenet')) # 参数有两种imagenet和None,None为从头开始训练,imagenet为从网络下载已训练好的模型开始训练
resnet50_fine_tune.add(tf.keras.layers.Dense(102, activation = 'softmax')) # 因为include_top = False,所以需要自己定义最后一层
resnet50_fine_tune.layers[0].trainable = False # 因为参数是从imagenet初始化的,所以我们可以只调整最后一层的参数

resnet50_fine_tune.compile(loss="categorical_crossentropy",
                           optimizer="sgd", metrics=['accuracy']) #对于微调-finetune来说,优化器使用sgd来说更好一些 
resnet50_fine_tune.summary()


def plot_learning_curves(history, label, epcohs, min_value, max_value):
    data = {}
    data[label] = history.history[label]
    data['val_'+label] = history.history['val_'+label]
    pd.DataFrame(data).plot(figsize=(8, 5))
    plt.grid(True)
    plt.axis([0, epochs, min_value, max_value])
    plt.show()



plot_learning_curves(history, 'accuracy', epochs, 0, 1)
plot_learning_curves(history, 'loss', epochs, 0, 2)

# 
resnet50 = tf.keras.applications.ResNet50(include_top = False,
                                       pooling = 'avg',
                                       weights = 'imagenet')
resnet50.summary()

# 
for layer in resnet50.layers[0:-5]:
    layer.trainable = False

resnet50_new = tf.keras.models.Sequential([
    resnet50,
    tf.keras.layers.Dense(102, activation = 'softmax'),
])
resnet50_new.compile(loss="categorical_crossentropy",
                     optimizer="sgd", metrics=['accuracy'])
resnet50_new.summary()
epochs = 10
# history = resnet50_fine_tune.fit_generator(train_generator,
 b 
