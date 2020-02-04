import random
from assignment01 import language_model as lm

choice = random.choice
classroom = '''
classroom = 时间 人物 位置 动作 设施
时间 = 早上| 中午 |晚上 |傍晚 | 某个时侯
某个时侯 = 8:30 | 10:30 | 12:00 | 14:00 | 17:30 | 18:00
人物 = 学生| 老师| 同学们 |老师们 | 某个具体的人
某个具体的人 = 小明| 小红| 小亮 
位置 = 在教室里| 在座位上| 在讲台边| 在课桌旁| 在门边
动作 = 擦 |洗| 拖 |拿着 |读 |讲 
设施 = 桌子 |椅子 |黑板 |扫帚 |拖把 |书| 课
'''

person_introduce = '''
person_introduce = 称呼 动作 地方 量词 职业
称呼 = 他是 |我是 |你是 |他们是 | 具体某人
具体某人 = 小明是 |小红是 |小亮是
动作 = 来自于| 居住在| 工作于 
地方 = 北京的 |上海的 |美国的 |成都的 |杭州的
量词 = 一名 |一位 | 省略
省略 = null
职业 = 工程师|设计师 |会计师| 建筑师 |程序员
'''


def create_grammar(grammar_str, split='=', line_split='\n'):
    grammar = {}
    for line in grammar_str.split(line_split):
        if not line.strip(): continue
        exp, stmt = line.split(split)
        grammar[exp.strip()] = [s.split() for s in stmt.split('|')]
    return grammar


def generate(gram, target):
    if target not in gram: return target  # means target is a terminal expression
    expanded = [generate(gram, t) for t in choice(gram[target])]
    return ''.join([e if e != '/n' else '\n' for e in expanded if e != 'null'])


def generate_n(num, gram, target):
    for i in range(num):
        print(generate(create_grammar(gram), target=target))


def generate_best():
    sentences = []
    for sen in [generate(create_grammar(classroom), target='classroom') for i in range(10)]:
        pro = lm.get_probability(sen)
        sentences.append((sen, pro))
        print('sentence: {} with Prb: {}'.format(sen, pro))
    sentences = sorted(sentences, key=lambda x: x[1], reverse=True)
    print('the best sentence is:{}'.format(sentences[0]))


generate_best()
