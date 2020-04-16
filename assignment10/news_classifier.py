#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/14 0014 23:54
# @Author  : honwaii
# @Email   : honwaii@126.com
# @File    : news_classifier.py
import os
from functools import reduce
import pickle
import gensim
import pandas as pd
import jieba
import numpy as  np
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB, GaussianNB


class News:
    def __init__(self, content: str, label: int):
        self.content = content
        self.label = label


def handle_news(stop_words_list: list):
    essays_path = './news_data.csv'
    contents = pd.read_csv(essays_path, encoding='gb18030', usecols=["source", "content"])
    news = []
    labels = []
    count = 0
    for each in contents.iterrows():
        content = str(each[1]['content']).strip()
        source = str(each[1]['source']).strip()
        if content == 'nan':
            continue
        if content is None or not isinstance(content, str):
            continue
        content = handle_doc(content, stop_words_list)
        news.append(content)
        if '新华社' in source:
            labels.append('1')
        else:
            labels.append('0')
        count += 1
        if count % 2000 == 0:
            print('handle docs: ' + str(count))

    with open("./news.txt", 'w', encoding='utf-8') as f:
        f.writelines(news)
        f.flush()
        f.close()
    with open("./labels.txt", 'w', encoding='utf-8') as f:
        f.writelines(labels)
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


def handle_doc(doc: str, stop_words_list: list):
    doc = doc.replace("\n", "。").strip()
    doc = doc.replace(r"\n", "。").strip()
    doc = doc.replace("\r", "。").strip()
    doc = doc.replace("\t", "。").strip()
    doc = doc.replace("新华社", "").strip()
    content = split_content(doc, stop_words_list) + "\n"
    return content


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


def load_word_vector_model(path: str, self_trained: bool):
    print("加载的词向量的路径: " + path)
    # 加载glove转换的模型: 保存的为文本形式
    if self_trained:
        word_embedding = gensim.models.Word2Vec.load(path)
    else:
        word_embedding = KeyedVectors.load_word2vec_format(path)
    print('load finished.')
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


def generate_doc_vector(doc: str, word_vec_model: Word2Vec):
    words = doc.split(" ")
    word_vec = np.zeros(word_vec_model.vector_size)
    for word in words:
        word_vec += get_word_vector(word, word_vec_model)
    word_vec = word_vec / len(words)
    return word_vec


def compute_docs_vec(docs: list, model):
    return np.row_stack([generate_doc_vector(doc, model) for doc in docs])


def load_docs_labels(model):
    with open('./news.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        docs = [str(line) for line in lines]
    with open('./labels.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        labels_str = str(lines[0]).strip()
        labels = [int(label) for label in labels_str]

    docs_vec = compute_docs_vec(docs, model)
    labels = np.asarray(labels)
    return docs_vec, labels


# # TODO 数据集 划分训练集和测试集 数据贴标签


# # 根据y分层抽样，测试数据占20%

def train_model():
    x, y = load_docs_labels(word_vec_model)
    print(x.shape)
    train_idx, test_idx = train_test_split(range(len(y)), test_size=0.2, stratify=y)
    train_x = x[train_idx, :]
    train_y = y[train_idx]
    test_x = x[test_idx, :]
    test_y = y[test_idx]
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    # model = GaussianNB()
    model.fit(train_x, train_y)
    print("Training set score: {:.3f}".format(model.score(train_x, train_y)))
    print("Test set score: {:.3f}".format(model.score(test_x, test_y)))
    y_pred = model.predict(test_x)
    t=eval_model(test_y, y_pred, np.asarray([0, 1]))
    print(t)
    return model


def predict(doc, model):
    doc_vec = generate_doc_vector(doc, word_vec_model)
    doc_vec = np.asarray(doc_vec).reshape(1, -1)
    y = model['lr'].predict(doc_vec)
    return y


# 计算各项评价指标
def eval_model(y_true, y_pred, labels):
    # 计算每个分类的Precision, Recall, f1, support
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred)
    # 计算总体的平均Precision, Recall, f1, support
    tot_p = np.average(p, weights=s)
    tot_r = np.average(r, weights=s)
    tot_f1 = np.average(f1, weights=s)
    tot_s = np.sum(s)
    res1 = pd.DataFrame({
        u'Label': labels,
        u'Precision': p,
        u'Recall': r,
        u'F1': f1,
        u'Support': s
    })
    res2 = pd.DataFrame({
        u'Label': [u'总体'],
        u'Precision': [tot_p],
        u'Recall': [tot_r],
        u'F1': [tot_f1],
        u'Support': [tot_s]
    })
    res2.index = [999]
    res = pd.concat([res1, res2])
    return res[[u'Label', u'Precision', u'Recall', u'F1', u'Support']]


def save_model(model, output_dir):
    model_file = os.path.join(output_dir, u'model.pkl')
    with open(model_file, 'wb') as outfile:
        pickle.dump({
            'y_encoder': np.asarray([0, 1]),
            'lr': model
        }, outfile)
    return


def load_model(path):
    with open(path + 'model.pkl', 'rb') as infile:
        lr_model = pickle.load(infile)
    return lr_model


stop_words = open(u'stopwords.txt', "r", encoding="utf-8").readlines()
stop_words_list = [line.strip() for line in stop_words]
# # handle_news(stop_words_list)
# word_vec_model = load_word_vector_model('./sgns.wiki.model', False)
word_vec_model = load_word_vector_model(path='./word_embedding_model_100', self_trained=True)
model = train_model()
# save_model(model, './')
with open('./news_demo.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    doc = reduce(lambda x, y: x + y, lines)
    doc = handle_doc(doc, stop_words_list)
    f.close()
model = load_model('./')
result = predict(doc, model)
print(result)
