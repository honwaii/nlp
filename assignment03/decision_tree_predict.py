#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2020/2/11 0011 21:11
# @Author  : honwaii
# @Email   : honwaii@126.com
# @File    : decision_tree_predict.py

from collections import Counter
from icecream import ic
import numpy as np
import pandas as pd

mock_data = {
    'gender': ['F', 'F', 'F', 'F', 'M', 'M', 'M'],
    'income': ['+10', '-10', '+10', '+10', '+10', '+10', '-10'],
    'family_number': [1, 2, 2, 1, 1, 1, 2],
    'bought': [1, 1, 1, 0, 0, 0, 1],
}


# 根据训练数据，生成决策树的模型
def decision_tree_model(training_data: pd.DataFrame, target: str):
    feature_tree = {}
    features = []
    while True:
        # print(training_data)
        # print(training_data.shape)
        f, v = find_the_optimal_spilter(training_data, target)
        if f is None:
            break
        training_data = training_data[training_data[f] != v]
        features.append(f)
        feature_tree[f] = v
        training_data = training_data.drop(columns=[f])
        if training_data.empty:
            break
    return features, feature_tree


dataset = pd.DataFrame.from_dict(mock_data)


def entropy(elements):
    counter = Counter(elements)
    probs = [counter[c] / len(elements) for c in set(elements)]
    return -sum(p * np.log(p) for p in probs)


# 返回特征和信息熵
def find_the_optimal_spilter(training_data: pd.DataFrame, target: str) -> str:
    x_fields = set(training_data.columns.tolist()) - {target}
    spliter = None
    min_entropy = float('inf')
    for f in x_fields:
        ic(f)
        values = set(training_data[f])
        # ic(values)
        for v in values:
            sub_spliter_1 = training_data[training_data[f] == v][target].tolist()
            ic(sub_spliter_1)
            # split by the current feature and one value

            entropy_1 = entropy(sub_spliter_1)
            ic(entropy_1)

            sub_spliter_2 = training_data[training_data[f] != v][target].tolist()
            # ic(sub_spliter_2)

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


def predicate(input_feature_values, target: str):
    feature_names, feature_values = decision_tree_model(dataset, target)
    print("需依次判断的特征为: ", feature_names)
    print("各个特征判断为True的值： ", feature_values)
    columns = dataset.columns.values.tolist()
    need_judge = {}
    for index, v in enumerate(input_feature_values):
        need_judge[columns[index]] = v
    print("待判断的各个特征的值: ", need_judge)
    for f in feature_names:
        if need_judge[f] == feature_values[f]:
            return True
    return False


result = predicate(["M", "-10", 1], target='bought')
print("\n判断的结果为: ", result)
