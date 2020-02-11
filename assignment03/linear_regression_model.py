#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2020/2/9 0009 15:18
# @Author  : honwaii
# @Email   : honwaii@126.com
# @File    : linear_regression_model.py.py


from sklearn import linear_model
import matplotlib.pyplot as plt
import random
import numpy as np

lr = linear_model.LinearRegression()


def assuming_function(x):
    return random.randint(1, 5) * x + random.randint(-2, 2)


def generate_data():
    X = np.random.random_integers(low=1, high=20, size=(40, 1)) * 10
    y = [assuming_function(x) for x in X]
    y = np.array(y)
    return X, y


def f(x, k, b):
    return k * x + b


def predict(x):
    X, y = generate_data()
    reg = lr.fit(X, y)
    reg.score(X.reshape(-1, 1), y)
    p = lr.predict(np.array([x]).reshape(1, -1))
    return X, y, p, reg.coef_, reg.intercept_


def draw_graph(x):
    X, y, predicted, k, b = predict(x)
    print('输入: ' + str(x) + "\n预测结果是: " + str(predicted[0][0]))
    fig, ax = plt.subplots()
    ax.scatter(X, y, c='b')
    ax.plot(X, f(X, k, b), color='red')
    ax.scatter(x, predicted, c='r')
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()


def result(x):
    draw_graph(x)
    return


result(120)
