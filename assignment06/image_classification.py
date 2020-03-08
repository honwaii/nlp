#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/1 0001 22:39
# @Author  : honwaii
# @Email   : honwaii@126.com
# @File    : image_classification.py

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import time

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# for i in range(1, 11):
#     plt.subplot(2, 5, i)
#     plt.imshow(x_train[i - 1])
#     plt.text(3, 10, str(y_train[i - 1]))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()
#
num_classes = 10
model_name = 'cifar10.h5'

tf.executing_eagerly()


def load_data():
    (images_train, labels_train), (images_test, labels_test) = cifar10.load_data()

    images_train = images_train.astype('float32') / 255
    images_test = images_test.astype('float32') / 255

    # Convert class vectors to binary class matrices.
    labels_train = keras.utils.to_categorical(labels_train, num_classes)
    labels_test = keras.utils.to_categorical(labels_test, num_classes)
    return images_train, labels_train, images_test, labels_test


images_train, labels_train, images_test, labels_test = load_data()
print(images_train.shape)
print(labels_train.shape)
print(images_test.shape)
print(labels_test.shape)
print(labels_test[0])


def classification_model(x_train):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
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

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)

    # train the model using RMSprop
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    hist = model.fit(x_train, y_train, epochs=40, shuffle=True)
    model.save(model_name)

    # evaluate
    loss, accuracy = model.evaluate(x_test, y_test)
    return


def training():
    return


def build_model1():
    mnist_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
        tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    mnist_model.summary()
    return mnist_model


def train(x_train, y_train, model):
    model = build_model1()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    model.fit(x_train, y_train, validation_split=0.1, shuffle=True, batch_size=1500, epochs=3)
    return model


def predict(test, model):
    test = np.asarray(test).reshape(1, 32, 32, 3)
    print(test)
    result = model.predict(np.asarray(test).reshape(1, 32, 32, 3))
    return result


images_train, labels_train, images_test, labels_test = load_data()
model = build_model1()
trained_model = train(images_train, labels_train, model)
result = predict(images_test[0], trained_model)

print(labels_test[0])
temp = result[0].tolist()
print(temp.index(max(temp)))
