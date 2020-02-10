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
    print((x1, x2, c))
    return c


def predict(x, k=3):
    most_similar_datas = sorted(knn_model(X, y), key=lambda xi: distance(xi[0], x))
    y_hats = [_y for x, _y in most_similar_datas[:k]]
    return np.mean(y_hats)


random_data = np.random.random((40, 2))
X = random_data[:, 0] * 2
y = [assuming_function(x) for x in X]

p1 = predict(0.4)
p2 = predict(0.8)
p3 = predict(1.2)
p4 = predict(1.6)
p5 = predict(2)

plt.scatter(X, y)
plt.scatter([0.4, 0.8, 1.2, 1.6, 2], [p1, p2, p3, p4, p5], c='r')
plt.show()
