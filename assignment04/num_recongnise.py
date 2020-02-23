#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2020/2/23 0023 15:11
# @Author  : honwaii
# @Email   : honwaii@126.com
# @File    : num_recongnise.py

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
print(digits.keys())
print(digits['target_names'])
print(digits['data'][1])
for i in range(1, 11):
    plt.subplot(2, 5, i)
    plt.imshow(digits.data[i - 1].reshape([8, 8]), cmap=plt.cm.gray_r)
    plt.text(3, 10, str(digits.target[i - 1]))
    plt.xticks([])
    plt.yticks([])
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25)
y_train[y_train < 5] = 0
y_train[y_train >= 5] = 1
y_test[y_test < 5] = 0
y_test[y_test >= 5] = 1
print(y_train)


# print(len(X_train))

def sigmoid(z):
    '''
    Compute the sigmoid of z
    Arguments: z -- a scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    '''
    s = 1. / (1 + np.exp(-1 * z))
    return s


def initialize_parameters(dim):
    '''
    Argument: dim -- size of the w vector

    Returns:
    w -- initialized vector of shape (dim,1)
    b -- initializaed scalar
    '''

    w = np.random.random((dim, 1))
    b = 1.0

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b


# print(initialize_parameters(5))


def propagate(w, b, X, Y):
    '''
    Implement the cost function and its gradient for the propagation

    Arguments:
    w - weights
    b - bias
    X - data
    Y - ground truth
    '''
    m = X.shape[1]
    A = sigmoid(w * m + b)
    print(A[0])
    cost = sum([y * np.log(a) + (1 - y) * np.log(1 - a) for a, y in zip(A, Y)])

    dw = np.dot(X, (A - Y)) / m
    db = [(a - y) / len(A) for a, y in zip(A, Y)]
    print(len(A))
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {'dw': dw,
             'db': db}
    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    '''
    This function optimize w and b by running a gradient descen algorithm

    Arguments:
    w - weights
    b - bias
    X - data
    Y - ground truth
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params - dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    '''

    costs = []

    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y)

        dw = grads['dw']
        db = grads['db']

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w, "b": b}

    grads = {"dw": dw, "db": db}

    return params, grads, costs


def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights
    b -- bias
    X -- data

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(w * m + b)

    for i in range(A.shape[0]):
        if i == 0:
            continue
        if i > 5:
            Y_prediction = True
        else:
            Y_prediction = False
        print(i)
    assert (Y_prediction.shape == (1, m))

    return Y_prediction


learning_rate = 1e-2

iteration_num = 2000


def model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate, print_cost):
    """
    Build the logistic regression model by calling all the functions you have implemented.
    Arguments:
    X_train - training set
    Y_train - training label
    X_test - test set
    Y_test - test label
    num_iteration - hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    d - dictionary should contain following information w,b,training_accuracy, test_accuracy,cost
    eg: d = {"w":w,
             "b":b,
             "training_accuracy": traing_accuracy,
             "test_accuracy":test_accuracy,
             "cost":cost}
    """
    dim = len(X_train[0])
    w, b = initialize_parameters(dim)
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, True)
    Y_predicte = predict(params['w'], params['b'], X_test)
    count = 0

    for y, y_hat in (Y_test, Y_predicte):
        result = y - y_hat
        if result == 0:
            continue
        else:
            count += 1
    print("验证的错误率为:" + str(count / len(y)))


model(X_train, Y_train=y_train, X_test=X_test, Y_test=y_test, num_iterations=200, learning_rate=0.01, print_cost=True)
