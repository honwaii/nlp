#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2020/2/9 0009 16:28
# @Author  : honwaii
# @Email   : honwaii@126.com
# @File    : knn_model.py

from scipy.spatial.distance import cosine
import numpy as np
import matplotlib.pyplot as plt
import random


def knn_model(X, y):
    return [(Xi, yi) for Xi, yi in zip(X, y)]


def distance(x1, x2):
    return cosine(x1, x2)


def assmuing_function(x):
    # 在我们的日常生活中是常见的
    # 体重 -> 高血压的概率
    # 收入 -> 买阿玛尼的概率
    # 其实都是一种潜在的函数关系 + 一个随机变化
    return 13.4 * x + 5 + random.randint(-5, 5)


random_data = np.random.random((20, 2))
X = random_data[:, 0]
y = [assmuing_function(x) for x in X]


def predict(x, k=5):
    most_similar_datas = sorted(knn_model(X, y), key=lambda xi: distance(xi[0], x))[:k]
    print(most_similar_datas)
    y_hats = [_y for x, _y in most_similar_datas]
    print(y_hats)
    return np.mean(y_hats)


myself_knn = knn_model(X, y)
p = predict(0.9)
plt.scatter(X, y)
plt.scatter([0.9], [p], c='r')
plt.show()
print(p)
