#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2020/2/11 0011 21:11
# @Author  : honwaii
# @Email   : honwaii@126.com
# @File    : decision_tree_model.py

from collections import Counter
from icecream import ic
import numpy as np
import pandas as pd

# mock_data = {
#     'gender': ['F', 'F', 'F', 'F', 'M', 'M', 'M', 'F'],
#     'stature': [180, 165, 172, 190, 175, 158, 162, 168],
#     'bought': [0, 1, 0, 0, 0, 1, 1, 1]
# }

mock_data = {
    'gender': ['F', 'F', 'F', 'F', 'M', 'M', 'M'],
    'income': ['+10', '-10', '+10', '+10', '+10', '+10', '-10'],
    'family_number': [1, 1, 2, 1, 1, 1, 2],
    # 'pet': [1, 1, 1, 0, 0, 0, 1],
    'bought': [1, 1, 1, 0, 0, 0, 1],
}


def decision_tree_model():
    return


dataset = pd.DataFrame.from_dict(mock_data)
sub_split_1 = dataset[dataset['family_number'] == 1]['bought'].tolist()
sub_split_2 = dataset[dataset['family_number'] != 1]['bought'].tolist()


print(dataset[dataset['family_number'] == 1])
print(sub_split_1)
print(sub_split_2)


def entropy(elements):
    counter = Counter(elements)
    probs = [counter[c] / len(elements) for c in set(elements)]
    ic(probs)
    return -sum(p * np.log(p) for p in probs)


def find_the_optimal_spilter(training_data: pd.DataFrame, target: str) -> str:
    x_fields = set(training_data.columns.tolist()) - {target}
    print(x_fields)
    spliter = None
    min_entropy = float('inf')

    for f in x_fields:
        ic(f)
        values = set(training_data[f])
        ic(values)
        for v in values:
            sub_spliter_1 = training_data[training_data[f] == v][target].tolist()
            ic(sub_split_1)
            # split by the current feature and one value

            entropy_1 = entropy(sub_spliter_1)
            ic(entropy_1)

            sub_spliter_2 = training_data[training_data[f] != v][target].tolist()
            ic(sub_split_2)

            entropy_2 = entropy(sub_spliter_2)
            ic(entropy_2)

            entropy_v = entropy_1 + entropy_2
            ic(entropy_v)

            if entropy_v <= min_entropy:
                min_entropy = entropy_v
                spliter = (f, v)

    print('spliter is: {}'.format(spliter))
    print('the min entropy is: {}'.format(min_entropy))
    return spliter


fm_n_1 = dataset[dataset['family_number'] == 1]
# fm_n_1[fm_n_1['income'] == '+10']
find_the_optimal_spilter(training_data=dataset, target='bought')
# find_the_optimal_spilter(dataset[dataset['family_number'] == 1], 'bought')
# print(fm_n_1[fm_n_1['income'] == '+10'])
# find_the_optimal_spilter(fm_n_1[fm_n_1['income'] == '+10'], 'bought')
# ic(entropy([1, 2, 3, 4]))
#
# # split_by_gender:
#
# print(entropy([1, 1, 1, 0]) + entropy([0, 0, 1]))
#
# # split_by_income:
# print(entropy([1, 1, 0, 0, 0]) + entropy([1, 1]))
#
# # split_by_family_number
# print(entropy([1, 1, 0, 0, 0]) + entropy([1, 1]))
#
# # 我们最希望找到一种feature， split_by_some_feature:
# # split_by_pet
# print(entropy([1, 1, 1, 1]) + entropy([0, 0, 0]))
