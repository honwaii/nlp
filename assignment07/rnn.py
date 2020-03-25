#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/22 0022 17:13
# @Author  : honwaii
# @Email   : honwaii@126.com
# @File    : rnn.py
import torch.nn as nn
import torch


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_num):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.hidden_num = hidden_num

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        for i in range(self.hidden_num):
            combined = torch.cat((input, hidden), 1)
            hidden = self.i2h(combined)

        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  # LSTM
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)
        return out

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
