#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/1 0001 22:39
# @Author  : honwaii
# @Email   : honwaii@126.com
# @File    : image_classification.py

import tensorflow.compat.v1 as tf

# import tensorflow as tf

tf.disable_v2_behavior()
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import os

# 指定第一块GPU可用
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定GPU的第二种方法

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'  # A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.3  # 定量
config.gpu_options.allow_growth = True  # 按需
# set_session(tf.compat.v1.Session(config=config))
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

num_classes = 10
model_name = 'cifar10.h5'


# tf.executing_eagerly()

def load_data():
    (images_train, labels_train), (images_test, labels_test) = cifar10.load_data()

    images_train = images_train.astype('float32') / 255
    images_test = images_test.astype('float32') / 255

    # Convert class vectors to binary class matrices.
    labels_train = keras.utils.to_categorical(labels_train, num_classes)
    labels_test = keras.utils.to_categorical(labels_test, num_classes)
    return images_train, labels_train, images_test, labels_test


def build_model(x_train):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.summary()
    return model


def train(x_train, y_train, model):
    opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    model.fit(x_train, y_train, validation_split=0.1, shuffle=True, batch_size=150, epochs=3)
    return model


def predict(test, model):
    test = np.asarray(test).reshape(1, 32, 32, 3)
    result = model.predict(np.asarray(test).reshape(1, 32, 32, 3))
    return result


images_train, labels_train, images_test, labels_test = load_data()
model = build_model(images_train)
trained_model = train(images_train, labels_train, model)
result = predict(images_test[0], trained_model)

print(labels_test[0])
temp = result[0].tolist()
print(temp.index(max(temp)))
