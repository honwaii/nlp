#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2020/2/13 0013 11:25
# @Author  : honwaii
# @Email   : honwaii@126.com
# @File    : k_means_model.py

import matplotlib.pyplot as plt
import random
from collections import defaultdict
from sklearn.cluster import KMeans


# 生成2-D待分类数据
def generate_clustering_data():
    X1 = [random.randint(0, 1000) for _ in range(0, 1000)]
    X2 = [random.randint(0, 1000) for _ in range(0, 1000)]
    return [[x1, x2] for x1, x2 in zip(X1, X2)]


def clustering(data):
    return KMeans(n_clusters=6, max_iter=1000).fit(data)


def draw_graph(cluster, locations):
    clustered_locations = defaultdict(list)
    colors = ['red', 'blue', 'orange', 'grey', 'black', 'yellow']
    # 数据所属分类和颜色标记可以在一个for循环内完成，不必像课程代码中一样分开。
    for label, location in zip(cluster.labels_, locations):
        clustered_locations[label].append(location)
        plt.scatter(*location, c=colors[label])
    # 标记聚类中心点的颜色，这里颜色不设定，其会自动随机选择颜色，不会撞色
    for center in cluster.cluster_centers_:
        plt.scatter(*center, s=100)
    plt.show()
    return


# 1. 生成待分类数据
clustering_data = generate_clustering_data()
# 2. 待分类数据的模型
cluster = clustering(clustering_data)
# 3. 绘制聚类数据图
draw_graph(cluster, clustering_data)
