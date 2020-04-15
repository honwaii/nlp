#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/14 0014 23:54
# @Author  : honwaii
# @Email   : honwaii@126.com
# @File    : news_classifier.py

import pandas as pd
import jieba
import numpy as  np
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec


class News:
    def __init__(self, content: str, label: int):
        self.content = content
        self.label = label


def handle_news():
    essays_path = './news_data.csv'
    contents = pd.read_csv(essays_path, encoding='gb18030', usecols=["source", "content"])
    stop_words_1 = open(u'哈工大停用词表.txt', "r", encoding="utf-8").readlines()
    stop_words_2 = open(u'中文停用词表.txt', "r", encoding="utf-8").readlines()
    stop_words = stop_words_1 + stop_words_2
    stop_words_list = [line.strip() for line in stop_words]
    news_xinhua = []
    news_other = []
    count = 0
    for each in contents.iterrows():
        content = str(each[1]['content']).strip()
        source = str(each[1]['source']).strip()
        if content == 'nan':
            continue
        if content is None or not isinstance(content, str):
            continue
        content = content.replace("\n", "。").strip()
        content = content.replace(r"\n", "。").strip()
        content = content.replace("\r", "。").strip()
        content = content.replace("\t", "。").strip()
        content1 = content.replace("新华社", "").strip()
        content = split_content(content1, stop_words_list) + "\n"
        if '新华社' in source:
            news_xinhua.append(content)
        else:
            news_other.append(content)
        count += 1
        if count % 1000 == 0:
            print('others:' + str(len(news_other)))
            print('新华社:' + str(len(news_xinhua)))
    with open("./新华社新闻.txt", 'w', encoding='utf-8') as f:
        f.writelines(news_xinhua)
        f.flush()
        f.close()
    with open("./other_news.txt", 'w', encoding='utf-8') as f:
        f.writelines(news_other)
        f.flush()
        f.close()
    # print("获取到的文章数:" + str(len(essays)))
    # print("新华社的文章数:" + str(count))
    return


def split_content(content: str, stop_words: list):
    simpled = ''
    s = content.replace("新华社", "")
    s = content.replace("\n", "")
    if s == "":
        return simpled
    segs = jieba.cut(s)
    for seg in segs:
        if seg in stop_words:
            continue
        simpled += seg + " "
    return simpled


def get_words_frequency_dict(path: str):
    print("load word frequency file from " + path)
    word2weight = {}
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if len(line) <= 0:
            continue
        line = line.split()
        if len(line) == 2:
            word2weight[line[0]] = float(line[1])
        else:
            print(line)
    return word2weight


def get_word_frequency(word_text, look_table):
    if word_text in look_table:
        return look_table[word_text]
    else:
        return 1.0


def get_word_vector(word: str, word_vector_model: Word2Vec):
    try:
        word_vector = word_vector_model[word]
    except KeyError:
        word_vector = np.zeros(word_vector_model.vector_size)
    return word_vector


def sentence_to_vec(sentence_list: list, word_vec: Word2Vec, look_table: dict, a=1e-3):
    sentence_set = []
    for sentence in sentence_list:
        vs = np.zeros(word_vec.vector_size)  # add all word2vec values into one vector for the sentence
        sentence_length = sentence.len()
        for word in sentence.word_list:
            a_value = a / (a + get_word_frequency(word, look_table))  # smooth inverse frequency, SIF
            vs = np.add(vs, np.multiply(a_value, get_word_vector(word, word_vec)))  # vs += sif * word_vector
        vs = np.divide(vs, sentence_length)  # weighted average
        sentence_set.append(vs)  # add to our existing re-calculated set of sentences
    if len(sentence_list) < 2:
        return sentence_set
    pca = PCA()
    pca.fit(np.array(sentence_set))
    u = pca.components_[0]  # the PCA vector
    u = np.multiply(u, np.transpose(u))  # u x uT

    if len(u) < word_vec.vector_size:
        for i in range(word_vec.vector_size - len(u)):
            u = np.append(u, 0)  # add needed extension for multiplication below

    sentence_vecs = []
    for vs in sentence_set:
        sub = np.multiply(u, vs)
        sentence_vecs.append(np.subtract(vs, sub))
    return sentence_vecs


def load_word_vector_model(path: str):
    print("加载的词向量的路径: " + path)
    # 加载glove转换的模型: 保存的为文本形式
    word_embedding = KeyedVectors.load_word2vec_format(path)
    return word_embedding


def get_max_length_doc(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        docs = str(lines).split("\\n")
        max_length = 0
        print(len(docs))
        for doc in docs:
            content = doc.split(" ")
            if len(content) > max_length:
                max_length = len(content)
    return max_length


def generate_sentence_vector(content: str, word_vec_model: Word2Vec):
    words = content.strip(" ")
    word_vec = np.zeros(word_vec_model.vector_size)
    for word in words:
        word_vec += get_word_vector(word, word_vec_model)
    word_vec = word_vec / len(words)
    return word_vec


def load_docs(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        docs = str(lines).split("\\n")
    return docs


# model = load_word_vector_model('./sgns.wiki.model')
# print('加载完成。')
# docs = load_docs('./other_news.txt')
# for i in range(1, 10):
#     vec = generate_sentence_vector(docs[i], model)
#     print(vec)

# TODO 数据集 划分训练集和测试集 数据贴标签
