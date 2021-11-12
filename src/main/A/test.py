# encoding:utf-8
'''
Created on 2015年10月25日
@author: Administrator
'''

import pandas  as pd
import re
import jieba
import nltk
import jieba.posseg as pseg
from gensim import corpora, models, similarities

# 导入自己添加的用户词语
# jieba.load_userdict(r'E:\PycharmProjects\untitled\new_trian.txt')
path = r"E:\PycharmProjects\untitled\src\resource\weibodata-train.txt"
df = pd.read_csv(path,delimiter="\t",header=None)
cont = df[0]
nwordall = []
for t in cont:
    words = pseg.cut(t)
    nword = ['']
    for w in words:
        if ((w.flag == 'n' or w.flag == 'v' or w.flag == 'a') and len(w.word) > 1):
            nword.append(w.word)
    nwordall.append(nword)
# 选择后的词生成字典
dictionary = corpora.Dictionary(nwordall)  # 用于生成字典类似与table，Counter模块中count
# print dictionary.token2id
# 生成语料库
corpus = [dictionary.doc2bow(text) for text in nwordall]
# tfidf加权
tfidf = models.TfidfModel(corpus)
# print tfidf.dfsx
# print tfidf.idf
corpus_tfidf = tfidf[corpus]

# 4. 主题模型lda，可用于降维
# lda流式数据建模计算，每块10000条记录，提取50个主题
lda = models.ldamodel.LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=50, update_every=1, chunksize=10000,
                               passes=1)

# 提取前面20个主题
for i in range(0, 20):
    print(lda.print_topics(i)[0])

test_doc = "须知写言情小说的女人都不是一般的战士，理想和现实的界线她们比谁都分得更清楚。 " \
           "安妮宝贝和她男人从认识到怀孕只用了半个月时间，总共见过三次面，勾搭、摆平、套牢 一气呵成，快、狠、准。	1"
test_doc = list(jieba.cut(test_doc))      #新文档进行分词
doc_bow = dictionary.doc2bow(test_doc)      #文档转换成bow
doc_lda = lda[doc_bow]                   #得到新文档的主题分布
#输出新文档的主题分布
print(doc_lda)
for topic in doc_lda:
    print( "%s\t%f\n"%(lda.print_topic(topic[0]), topic[1]))


# lda全部数据建模，提取100个主题
# lda = models.ldamodel.LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=100, update_every=0, passes=20)
# 利用原模型预测新文本主题
# doc_lda = lda[corpus_tfidf]





