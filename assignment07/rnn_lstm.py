#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/25 0025 23:24
# @Author  : honwaii
# @Email   : honwaii@126.com
# @File    : rnn_lstm.py

# Instantiate model
from datetime import time

import torch
from torch import nn

from assignment07.rnn import LSTM
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np

input_size = 57
hidden_size = 128
num_layers = 1
num_classes = 18
batch_size = 200


class CustomDataset():
    '''Creates a custom dataset from numpy arrays of features and labels
    for feeding into PyTorch dataloaders '''

    def __init__(self, features, labels):
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels).type(torch.LongTensor)
        self.len = len(features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return self.len


# Function to convert each line = each name into a one-hot tensor

def letter_to_index(letter):
    return all_letters.find(letter)


def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for i, letter in enumerate(line):
        tensor[i][0][letter_to_index(letter)] = 1
    return tensor


line_list_clean = [unicode_to_ascii(line) for line in line_list]
line_list_tensorized = [line_to_tensor(line) for line in line_list_clean]
# Pad sequences - automatically converts the data to a tensor
line_tensor = pad_sequence(line_list_tensorized)

# Transpose tensor to the correct format: Number of observations, 1, sequence length, features
line_tensor_permuted = line_tensor.permute(1, 2, 0, 3)
# Convert feature tensor to array
X = line_tensor_permuted.numpy()

# Map languages to integer labels
category_dict = dict(zip(all_categories, range(n_categories)))

# Convert all languages in the category list to an array of labels
category_list_numeric = [category_dict.get(i) for i in category_list]
y = np.array(category_list_numeric)

data_train = CustomDataset(features=X, labels=y)
data_test = CustomDataset(features=X, labels=y)

trainloader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)

model = LSTM(input_size, hidden_size, num_layers, num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)
# Send model to device
model.to(torch.device)

# Traing hyper-parameters
sequence_length = 10  # length of padded tensors
num_epochs = 40
learning_rate = 0.001

# Loss and optimizer
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Print out training progress every {print_every} batches
print_every = 200

# Code for printouts and visualizations ----------------

start_time = time.time()

print("Start of traing -- Device: {} -- Epochs: {} -- Batches: {} -- Batch size: {}"
      .format(device, num_epochs, len(trainloader), batch_size))

# Initiate variables and lists for training progress printouts and visualizations
running_loss = 0
running_total = 0
running_correct = 0
loss_list = []
loss_list_print_every = []

# Code for actual training -----------------------------

# Set model to training mode
model.train()

# Train the model
for epoch in range(num_epochs):
    for i, (lines, labels) in enumerate(trainloader):

        # Send data to GPU
        lines, labels = lines.to(device), labels.to(device)

        # Reshape lines batch
        lines = lines.reshape(-1, sequence_length, input_size)

        # Forward and backward pass
        output = model(lines)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Code for printouts and visualizations ----------------

        # Store running loss, total, correct
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        running_total += labels.size(0)
        running_correct += (predicted == labels).sum().item()
        loss_list.append(loss.item())

        # Print out  average training loss and accuracy every {print_every} batches
        if (i + 1) % print_every == 0:
            print("Epoch: {}/{} -- Batches: {}/{} -- Training loss: {:.3f} -- Training accuracy: {:.3f}"
                  .format(epoch + 1, num_epochs, i + 1, len(trainloader),
                          running_loss / print_every, running_correct / running_total))

            # Store running loss in list
            loss_list_print_every.append(running_loss / print_every)

            # Reset running loss and accuracy
            running_loss = 0
            running_total = 0
            running_correct = 0

print("Training complete. Total training time: {:.1f} seconds".format(time.time() - start_time))
