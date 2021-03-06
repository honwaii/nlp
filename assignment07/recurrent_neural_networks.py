#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/17 0017 23:32
# @Author  : honwaii
# @Email   : honwaii@126.com
# @File    : recurrent_neural_networks.py
from io import open
import glob
import os
import matplotlib.pyplot as plt
import random
import unicodedata
import string
import torch
import time
import math
from assignment07.rnn import LSTM, Multi_Layers_RNN, GRU
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


def find_files(path): return glob.glob(path)


all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicode_2_Ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Build the category_lines dictionary, a list of names per language
# Read a file and split into lines
def read_lines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode_2_Ascii(line) for line in lines]


def get_languages_and_names() -> (list, dict):
    category_lines = {}
    all_categories = []
    for filename in find_files('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = read_lines(filename)
        category_lines[category] = lines
    return all_categories, category_lines


# Find letter index from all_letters, e.g. "a" = 0
def letter_to_index(letter):
    return all_letters.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letter_to_tensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter_to_index(letter)] = 1
    return tensor


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor


def category_from_output(output, all_categories: list):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def sample(l):
    return l[random.randint(0, len(l) - 1)]


def sample_training(all_categories: list, category_lines: dict):
    category = sample(all_categories)  # all_categories:所有的国家名 ，category: 某个国家名
    line = sample(category_lines[category])  # line : 某个国家的人的名字
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    line_tensor = pad_sequence(line_tensor)
    return category, line, category_tensor, line_tensor


def sample_training1(all_categories: list, category_lines: dict):
    category = sample(all_categories)  # all_categories:所有的国家名 ，category: 某个国家名
    line = sample(category_lines[category])  # line : 某个国家的人的名字
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor


learning_rate = 0.005  # If you set this too high, it might explode. If too low, it might not learn
input_size = n_letters


def multi_layers_train(category_tensor, line_tensor, rnn: Multi_Layers_RNN, criterion):
    hidden = rnn.initHidden()

    rnn.zero_grad()
    output = None
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)  # 第i个字母的tensor
    # 将所有的输入传入后，最后得到的输出，来比较loss
    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()


# Keep track of losses for plotting

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def training(all_categories: list, category_lines: dict, rnn, n_iters):
    start = time.time()
    current_loss = 0
    all_losses = []
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()

    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = sample_training(all_categories, category_lines)
        output, loss = lstm_gru_train(category_tensor, line_tensor, rnn, criterion, optimizer)
        current_loss += loss

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = category_from_output(output, all_categories)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (
                iter, iter / n_iters * 100, time_since(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0
    return all_losses


def print_result(iter, output, category, start, loss, line, all_losses, current_loss):
    return current_loss


def muti_layers_training(all_categories: list, category_lines: dict, rnn):
    start = time.time()
    current_loss = 0
    all_losses = []
    criterion = nn.CrossEntropyLoss()

    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = sample_training1(all_categories, category_lines)
        output, loss = multi_layers_train(category_tensor, line_tensor, rnn, criterion)
        current_loss += loss

        if iter % print_every == 0:
            guess, guess_i = category_from_output(output, all_categories)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (
                iter, iter / n_iters * 100, time_since(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    return all_losses


def lstm_gru_train(category_tensor, line_tensor, rnn: LSTM, criterion, optimizer):
    output = rnn(line_tensor)
    loss = criterion(output, category_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return output, loss.item()


def evaluate(line_tensor, rnn: Multi_Layers_RNN):
    hidden = rnn.initHidden()
    output = None
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output


def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(line_to_tensor(input_line), rnn)

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])


def plot_loss(all_losses: list, labels: list):
    for index, each in enumerate(all_losses):
        plt.plot([x for x in range(1, len(each) + 1)], each, label=labels[index])
    plt.legend()
    plt.show()
    return


def max_name_length(category_lines: dict):
    return len(max(category_lines, key=len))


n_hidden = 128
n_iters = 1000  # 这个数字你可以调大一些
print_every = 100
plot_every = 100
all_categories, category_lines = get_languages_and_names()
sequence_length = max_name_length(category_lines)


def diff_hidden_layers():
    rnn_multi_layers = Multi_Layers_RNN(n_letters, n_hidden, len(all_categories), 1)
    all_losses_1 = training(all_categories, category_lines, rnn_multi_layers, n_iters)
    plt.plot([x for x in range(1, len(all_losses_1) + 1)], all_losses_1, color='r')
    plt.show()
    return


def diff_model_train():
    all_losses = []
    labels = []
    n_iters = 1000
    rnn_lstm = LSTM(n_letters, n_hidden, 1, len(all_categories))
    losses = training(all_categories, category_lines, rnn_lstm, n_iters)
    all_losses.append(losses)
    labels.append('lstm model, n_iters=1000')

    gru_lstm = GRU(n_letters, n_hidden, 1, len(all_categories))
    losses = training(all_categories, category_lines, gru_lstm)
    all_losses.append(losses)
    labels.append('gru model, n_iters=1000')

    n_iters = 5000
    rnn_lstm = LSTM(n_letters, n_hidden, 1, len(all_categories))
    losses = training(all_categories, category_lines, rnn_lstm, n_iters)
    all_losses.append(losses)
    labels.append('lstm model, n_iters=1000')

    gru_lstm = GRU(n_letters, n_hidden, 1, len(all_categories))
    losses = training(all_categories, category_lines, gru_lstm)
    all_losses.append(losses)
    labels.append('gru model, n_iters=1000')

    plot_loss(all_losses, labels)


def diff_model_diff_layers():
    all_losses = []
    labels = []
    n_iters = 1000
    rnn_lstm_1 = LSTM(n_letters, n_hidden, 1, len(all_categories))
    rnn_lstm_2 = LSTM(n_letters, n_hidden, 2, len(all_categories))
    rnn_lstm_3 = LSTM(n_letters, n_hidden, 3, len(all_categories))

    all_losses_1 = training(all_categories, category_lines, rnn_lstm_1, n_iters)
    all_losses.append(all_losses_1)
    labels.append('rnn_lstm with 1 hidden layer')
    all_losses_2 = training(all_categories, category_lines, rnn_lstm_2, n_iters)
    all_losses.append(all_losses_2)
    labels.append('rnn_lstm with 2 hidden layer')
    all_losses_3 = training(all_categories, category_lines, rnn_lstm_3, n_iters)
    all_losses.append(all_losses_3)
    labels.append('rnn_lstm with 3 hidden layer')

    gru_1 = GRU(n_letters, n_hidden, 1, len(all_categories))
    gru_2 = GRU(n_letters, n_hidden, 2, len(all_categories))
    gru_3 = GRU(n_letters, n_hidden, 3, len(all_categories))
    all_losses_1 = training(all_categories, category_lines, gru_1, n_iters)
    all_losses.append(all_losses_1)
    labels.append('rnn_gru with 1 hidden layer')
    all_losses_2 = training(all_categories, category_lines, gru_2, n_iters)
    all_losses.append(all_losses_2)
    labels.append('rnn_gru with 2 hidden layer')
    all_losses_3 = training(all_categories, category_lines, gru_3, n_iters)
    all_losses.append(all_losses_3)
    labels.append('rnn_gru with 3 hidden layer')
    plot_loss(all_losses, labels)


def multi_layers_model():
    hidden_num = 3
    all_losses = []
    labels = []
    for layer_num in range(0, hidden_num):
        rnn_multi_layers = Multi_Layers_RNN(n_letters, n_hidden, len(all_categories), layer_num + 1)
        losses = muti_layers_training(all_categories, category_lines, rnn_multi_layers)
        all_losses.append(losses)
        labels.append('layer_num = ' + str(layer_num + 1))
    plot_loss(all_losses, labels)
    return


def training2(all_categories: list, category_lines: dict, rnn, n_iters):
    start = time.time()
    current_loss = 0
    all_losses = []
    criterion = nn.NLLLoss()

    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = sample_training(all_categories, category_lines)
        output, loss = lstm_gru_train(category_tensor, line_tensor, rnn, criterion, optimizer)
        current_loss += loss

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = category_from_output(output, all_categories)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (
                iter, iter / n_iters * 100, time_since(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0
    return all_losses


def train_diff_loss():
    all_losses = []
    labels = []
    n_iters = 1000
    rnn_lstm_1 = LSTM(n_letters, n_hidden, 1, len(all_categories))
    losses = training(all_categories, category_lines, rnn_lstm_1, n_iters)
    all_losses.append(losses)
    labels.append('lstm model, criterion = nn.CrossEntropyLoss')

    rnn_lstm_2 = LSTM(n_letters, n_hidden, 1, len(all_categories))
    losses = training2(all_categories, category_lines, rnn_lstm_2, n_iters)
    all_losses.append(losses)
    labels.append('lstm model, criterion = nn.NLLLoss')

    plot_loss(all_losses, labels)


# diff_hidden_layers()

# predict('Dovesky')
# predict('Jackson')
# predict('Satoshi')

# 代码练习
# 1. 尝试在我们的RNN模型中添加更多layers，然后观察Loss变化
multi_layers_model()
#
# 2. 将原始的RNN模型改成nn.LSTM和nn.GRU， 并且改变 n_iters = 1000 这个值，观察其变化
# todo
diff_model_train()
# 3. 把该RNN模型变成多层RNN模型，观察Loss的变化
# todo
diff_model_diff_layers()
#
# 4. Pytorch里边常用nn.NLLoss来代替crossentropy，将criterion改为nn.NLLoss，观察变化
# todo
train_diff_loss()

