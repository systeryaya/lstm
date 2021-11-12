


# 5. word2vec 词向量化，可用于比较词相似度，寻找对应关系，词聚类
from gensim import models
import pandas as pd

path = r"E:\PycharmProjects\untitled\src\resource\new_trian.txt"
df = pd.read_csv(path,delimiter="\t",header=None)
nwordall = df[0]
sentences = models.word2vec.LineSentence(nwordall)
# size为词向量维度数,windows窗口范围,min_count频数小于5的词忽略,workers是线程数
model = models.word2vec.Word2Vec(nwordall, size=100, window=5, min_count=5, workers=4)
model.save("model.txt")
print(model[u'忠诚'])
# 向量表示
sim = model.most_similar(positive=[u'忠诚', u'珍惜'])
# 相近词
for s in sim:
    print ("word:%s,similar:%s " % (s[0], s[1]))