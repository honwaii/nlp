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
    def __init__(self, input_size, hidden_size, hidden_num, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=hidden_num,
                                 batch_first=True)
        self.out = nn.Linear(input_size + hidden_size, output_size)

    def forward(self, input, hidden):
        output, (hn, cn) = self.rnn(input, None)
        output = self.out(output)
        return output

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
