
import codecs
import jieba
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim import corpora, models, similarities
import pandas as pd
import jieba.posseg as psg

path = r"E:\PycharmProjects\untitled\src\resource\weibodata-train.txt"
df = pd.read_csv(path,delimiter="\t",header=None)
train = df[0]

stop_path = r"E:\PycharmProjects\untitled\src\resource\stopWord_chinese.txt"
stopwords = codecs.open(stop_path,'r',encoding='gbk').readlines()
stopwords = [ w.strip() for w in stopwords ]

#词性过滤，去停用词
train_set = []
for line in train:
    words = psg.cut(line)
    nword = []
    for w in words:
        if ((w.flag == 'n' or w.flag == 'v' or w.flag == 'a') and len(w.word) > 1) and w.word not in stopwords:
            nword.append(w.word)
    train_set.append(nword)

# 选择后的词生成字典
dictionary = Dictionary(train_set)
# 构建训练语料
corpus = [ dictionary.doc2bow(text) for text in train_set]
# tfidf加权
tfidf = models.TfidfModel(corpus)
corpusTfidf = tfidf[corpus]

# 4. 主题模型lda，可用于降维
# lda流式数据建模计算，每块10000条记录，提取50个主题
lda = models.ldamodel.LdaModel(corpus=corpusTfidf, id2word=dictionary, num_topics=50, update_every=1, chunksize=10000,
                               passes=1)

# 提取前面20个主题
for i in range(0, 20):
    print(lda.print_topics(i)[0])