import pandas as pd
import re
import jieba as jieba
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon

filename = 'C:\\Users\\honwa\\Desktop\\movie_comments.csv'

cleaned_data_file = 'C:\\Users\\honwa\\Desktop\\comments.txt'


#  清洗数据
def clean_data_source(filename):
    chunk_size = 500
    chunks = []
    content = pd.read_csv(filename, encoding='utf-8', usecols=['comment'], iterator=True)
    loop = True
    index = 0
    while loop:
        try:
            chunk = content.get_chunk(chunk_size)
            start_index = index * chunk_size
            for i in range(chunk.size):
                chunks.append(chunk.at[start_index + i, "comment"])
            index += 1
        except StopIteration:
            print("read finish.")
            loop = False
    return combine_comments(chunks)


def combine_comments(comments):
    temp = []
    for line in comments:
        s = re.findall('\w+', str(line))
        temp = temp + s
    return temp


# 保存清洗后的所有文本
def save_cleaned_data(contents, cleaned_data_file):
    with open(cleaned_data_file, 'w', encoding="utf-8") as f:
        for a in contents:
            f.write(a)


#  读取已保存的清洗后的数据
def get_cleaned_data(cleaned_data_file):
    with open(cleaned_data_file, 'r', encoding="utf-8") as f:
        return f.read()


def plot_statistic(words_count):
    x = [i for i in range(100)]
    frequencies = [f for w, f in words_count.most_common(100)]
    plt.plot(x, frequencies)
    plt.show()
    plt.plot(x, np.log(frequencies))
    plt.show()


def get_all_keys(word_count):
    return list(word_count.keys())


def get_2_gram(words_count):
    token = get_all_keys(words_count)
    return [''.join(token[i:i + 2]) for i in range(len(token[:-2]))]


def prob_1(word):
    return words_count[word] / sum(words_count.values())


def prob_2(word1, word2, words_count):
    token2 = get_2_gram(words_count)
    words_count_2 = Counter(token2)
    if word1 + word2 in words_count_2:
        return words_count_2[word1 + word2] / len(token2)
    else:
        return 1 / len(token2)


def get_probability(sentence):
    words = list(jieba.cut(sentence))
    sentence_pro = 1
    for i, word in enumerate(words[:-1]):
        next_ = words[i + 1]
        probability = prob_2(word, next_, words_count)
        sentence_pro *= probability
    return sentence_pro


# 2. 使用新数据源完成语言模型的训练
# 2.1 清洗数据，返回所有清洗后的文本
# contents = clean_data_source(filename)
# 已清洗过并有保存时，可直接从文件中读取
contents = get_cleaned_data(cleaned_data_file)
# 2.2 对文本进行切词
words_count = Counter(jieba.cut(get_cleaned_data(cleaned_data_file)))
# plot_statistic(words_count)
