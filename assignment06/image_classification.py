#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/1 0001 22:39
# @Author  : honwaii
# @Email   : honwaii@126.com
# @File    : image_classification.py

import tensorflow as tf
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

for i in range(1, 11):
    plt.subplot(2, 5, i)
    plt.imshow(x_train[i - 1])
    plt.text(3, 10, str(y_train[i - 1]))
    plt.xticks([])
    plt.yticks([])
plt.show()
