#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/14 0014 23:54
# @Author  : honwaii
# @Email   : honwaii@126.com
# @File    : news_classifier.py

import pandas as pd
import jieba
import time


class News:
    def __init__(self, content: str, label: int):
        self.content = content
        self.label = label


def handle_news():
    essays_path = './news_data.csv'
    contents = pd.read_csv(essays_path, encoding='gb18030', usecols=["source", "content"])
    essays = []
    count = 0
    stop_words = open(u'中文停用词表.txt', "r", encoding="utf-8").readlines()
    stop_words_list = [line.strip() for line in stop_words]
    for each in contents.iterrows():
        content = str(each[1]['content']).strip()
        source = str(each[1]['source']).strip()
        if content is None or not isinstance(content, str):
            continue
        content = split_content(content, stop_words_list)
        if '新华社' in source:
            news = News(content, 1)
            count += 1
        else:
            news = News(content, 0)
        essays.append(news)
        if count % 10000 == 0:
            print(content)
    print("获取到的文章数:" + str(len(essays)))
    print("新华社的文章数:" + str(count))
    return essays


def split_content(content: str, stop_words: list):
    simpled = ''
    s = content.replace("\n", "")
    if s == "":
        return simpled
    segs = jieba.cut(s)
    for seg in segs:
        if seg in stop_words:
            continue
        simpled += seg + " "
    return simpled


handle_news()
