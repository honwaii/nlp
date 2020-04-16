#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/14 0014 23:54
# @Author  : honwaii
# @Email   : honwaii@126.com
# @File    : news_classifier.py
import gensim
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
    stop_words = open(u'stopwords.txt', "r", encoding="utf-8").readlines()
    stop_words_list = [line.strip() for line in stop_words]
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
        content = content.replace("\n", "。").strip()
        content = content.replace(r"\n", "。").strip()
        content = content.replace("\r", "。").strip()
        content = content.replace("\t", "。").strip()
        content1 = content.replace("新华社", "").strip()
        content = split_content(content1, stop_words_list) + "\n"
        news.append(content)
        if '新华社' in source:
            labels.append('1')
        else:
            labels.append('0')
        count += 1
        if count % 1000 == 0:
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
    # word_embedding = KeyedVectors.load_word2vec_format
    word_embedding = gensim.models.Word2Vec.load(path)
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
    words = doc.strip(" ")
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
    train_idx, test_idx = train_test_split(range(len(y)), test_size=0.2, stratify=y)
    train_x = x[train_idx, :]
    train_y = y[train_idx]
    test_x = x[test_idx, :]
    test_y = y[test_idx]
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    model.fit(train_x, train_y)
    return model


def predict(doc: str):
    doc_vec = generate_doc_vector(doc, word_vec_model)
    result = model.predict(doc_vec)
    return result


word_vec_model = load_word_vector_model('./word_embedding_model_100')
# handle_news()
