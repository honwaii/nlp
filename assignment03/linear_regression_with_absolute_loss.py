#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2020/2/16 0016 21:51
# @Author  : honwaii
# @Email   : honwaii@126.com
# @File    : linear_regression_with_absolute_loss.py

from sklearn.datasets import load_boston
import random
import matplotlib.pyplot as plt

dataset = load_boston()
x, y = dataset['data'], dataset['target']
X_rm = x[:, 5]
plt.scatter(X_rm, y)
plt.show()


# Gradient descent

def price(rm, k, b):
    return k * rm + b


def loss(y, y_hat):
    # return sum((y_i - y_hat_i) ** 2 for y_i, y_hat_i in zip(list(y), list(y_hat))) / len(list(y))
    # loss 定义为绝对值
    return sum(abs(y_i - y_hat_i) for y_i, y_hat_i in zip(list(y), list(y_hat))) / len(list(y))


# define partial derivative
def partial_derivative_k(x, y, y_hat):
    n = len(y)
    gradient = 0
    for x_i, y_i, y_hat_i in zip(list(x), list(y), list(y_hat)):
        if y_i >= y_hat_i:
            gradient += (-x_i)
        else:
            gradient += x_i
    return gradient / n


def partial_derivative_b(y, y_hat):
    n = len(y)
    gradient = 0
    for y_i, y_hat_i in zip(list(y), list(y_hat)):
        if y_i >= y_hat_i:
            gradient += -1
        else:
            gradient += 1
    return gradient / n


def train():
    # initialized parameters
    k = random.random() * 200 - 100  # -100 100
    b = random.random() * 200 - 100  # -100 100

    learning_rate = 1e-2

    iteration_num = 2000
    losses = []
    for i in range(iteration_num):
        price_use_current_parameters = [price(r, k, b) for r in X_rm]  # \hat{y}
        current_loss = loss(y, price_use_current_parameters)
        losses.append(current_loss)
        if i % 100 == 0:
            print("Iteration {}, the loss is {}, parameters k is {} and b is {}".format(i, current_loss, k, b))

        k_gradient = partial_derivative_k(X_rm, y, price_use_current_parameters)
        b_gradient = partial_derivative_b(y, price_use_current_parameters)

        k = k + (-1 * k_gradient) * learning_rate
        b = b + (-1 * b_gradient) * learning_rate
    plt.plot(list(range(iteration_num)), losses)
    plt.show()
    return k, b


best_k, best_b = train()
price_use_best_parameters = [price(r, best_k, best_b) for r in X_rm]
plt.scatter(X_rm, price_use_best_parameters)
plt.scatter(X_rm, y)
plt.show()
