#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2020/2/9 0009 16:28
# @Author  : honwaii
# @Email   : honwaii@126.com
# @File    : knn_model.py

import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.spatial.distance import cosine


def assuming_function(x):
    return 10 * x + 3 + random.randint(-5, 5)


# 1. 将数据存到内存中
def knn_model(X, y):
    return [(Xi, yi) for Xi, yi in zip(X, y)]


# 2. 计算各点与预测点的余弦相似度
# 源码中的余弦相似度的计算为啥是1-cos(theta)
def distance(x1, x2):
    c = cosine(x1, x2)
    return c


def draw_graph(x, y):
    plt.scatter(x, y, c='r')
    plt.show()
    return


def result(x, k=3):
    X, Y = generate_training_data()
    most_similar_datas = sorted(knn_model(X, Y), key=lambda xi: distance(xi[0], x))
    y_hats = [_y for x, _y in most_similar_datas[:k]]
    predicted = np.mean(y_hats)
    draw_graph(x, predicted)
    return


def generate_training_data():
    random_data = np.random.random((40, 2))
    X = random_data[:, 0] * 2
    Y = [assuming_function(x) for x in X]
    plt.scatter(X, Y)
    return X, Y


result(1)
