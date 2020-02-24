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


# for i in range(1, 11):
#     plt.subplot(2, 5, i)
#     plt.imshow(digits.data[i - 1].reshape([8, 8]), cmap=plt.cm.gray_r)
#     plt.text(3, 10, str(digits.target[i - 1]))
#     plt.xticks([])
#     plt.yticks([])
# # plt.show()


def generate_dataset():
    digits = datasets.load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25)
    y_train[y_train < 5] = 0
    y_train[y_train >= 5] = 1
    y_test[y_test < 5] = 0
    y_test[y_test >= 5] = 1
    y_train = y_train.reshape(y_train.shape[0], -1)
    return X_train, X_test, y_train, y_test


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

    w = np.zeros(shape=(dim, 1))
    b = 0

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
    m = X.shape[0]
    A = sigmoid(np.dot(X, w) + b)

    cost = -1. / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    dw = (1.0 / m) * np.dot(X.T, (A - Y))
    db = (1. / m) * np.sum(A - Y)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {'dw': dw, 'db': db}
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
    m = X.shape[0]
    Y_prediction = np.zeros((1, m))

    A = sigmoid(w * m + b)
    # print(A.shape)
    for i in range(A.shape[0]):
        Y_prediction[0, i] = 1 if A[i, 0] > 0.5 else 0
    assert (Y_prediction.shape == (1, m))

    return Y_prediction


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

    dim = X_train.shape[1]
    # print("-->" + str(dim))
    w, b = initialize_parameters(dim)
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    Y_predicate = predict(params['w'], params['b'], X_test)

    return Y_predicate, costs


def draw_diff_learning_rate():
    X_train, X_test, y_train, y_test = generate_dataset()
    y_predicte, costs_1 = model(X_train, Y_train=y_train, X_test=X_test, Y_test=y_test, num_iterations=2000,
                                learning_rate=0.001,
                                print_cost=False)
    y_predicte, costs_2 = model(X_train, Y_train=y_train, X_test=X_test, Y_test=y_test, num_iterations=2000,
                                learning_rate=0.005,
                                print_cost=False)
    y_predicte, costs_3 = model(X_train, Y_train=y_train, X_test=X_test, Y_test=y_test, num_iterations=2000,
                                learning_rate=0.01,
                                print_cost=False)
    plt.plot([x for x in range(len(costs_1))], costs_1, 'c*-')
    plt.plot([x for x in range(len(costs_2))], costs_2, 'g*-')
    plt.plot([x for x in range(len(costs_3))], costs_3, 'm*-')
    plt.show()


draw_diff_learning_rate()
