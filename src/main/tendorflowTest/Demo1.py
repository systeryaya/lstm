# coding=utf-8
import numpy as np
import tensorflow as tf
import random
from collections import Counter
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

'''运行无误'''

"""
'I'm super man'
tokenize:
['I', ''m', 'super','man' ]
"""


"""
词形还原(lemmatizer)，即把一个任何形式的英语单词还原到一般形式，与词根还原不同(stemmer)，后者是抽取一个单词的词根。
"""

path = r"E:\PycharmProjects\untitled\src\main\resource"
pos_file = path + r'\pos1.txt'
neg_file = path + r'\neg1.txt'



# 创建词汇表
def create_lexicon(pos_file, neg_file):
    lex = []
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
    stops = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    # 读取文件
    def process_file(f):
        with open(pos_file, 'r') as f:
            lex = []
            lines = f.readlines()
            # print(lines)
            for line in lines:
                # word = re.sub("[^a-zA-Z]", " ", line.lower()).split()    按空格分词，且只保留字母，大写转小写，数字符号都不要
                text_list = word_tokenize(line.lower())  #分词，大写转小写
                # 去掉标点符号
                text_list = [word for word in text_list if word not in english_punctuations]
                # 去掉停用词
                text_list = [word for word in text_list if word not in stops]
                # 词性还原
                text_list = [lemmatizer.lemmatize(word) for word in text_list]
                lex += text_list
            return lex

    lex += process_file(pos_file)
    lex += process_file(neg_file)
    # print(len(lex))
    # lemmatizer = WordNetLemmatizer()
    # lex = [lemmatizer.lemmatize(word) for word in lex]  # 词形还原 (cats->cat)

    # nltk.pos_tag(text_list)  词性标注

    # 统计每个词出现的次数
    word_count = Counter(lex)
    # print(word_count)
    # {'.': 13944, ',': 10536, 'the': 10120, 'a': 9444, 'and': 7108, 'of': 6624, 'it': 4748, 'to': 3940......}
    # 去掉一些常用词,像the,a and等等，和一些不常用词; 这些词对判断一个评论是正面还是负面没有做任何贡献
    #过滤低词频
    lex = []
    for word in word_count:
        if word_count[word] < 2000 and word_count[word] > 20:  # 这写死了，好像能用百分比
            lex.append(word)  # 齐普夫定律-使用Python验证文本的Zipf分布 http://blog.topspeedsnail.com/archives/9546
    return lex


lex = create_lexicon(pos_file, neg_file)


# lex里保存了文本中出现过的单词。

# 把每条评论转换为向量, 转换原理：
# 假设lex为['woman', 'great', 'feel', 'actually', 'looking', 'latest', 'seen', 'is'] 当然实际上要大的多
# 评论'i think this movie is great' 转换为 [0,1,0,0,0,0,0,1], 把评论中出现的字在lex中标记，出现过的标记为1，其余标记为0
def normalize_dataset(lex):
    dataset = []

    # lex:词汇表；review:评论；clf:评论对应的分类，[0,1]代表负面评论 [1,0]代表正面评论
    def string_to_vector(lex, review, clf):
        words = word_tokenize(review.lower())
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        features = np.zeros(len(lex))
        for word in words:
            if word in lex:
                features[lex.index(word)] = 1  # 一个句子中某个词可能出现两次,可以用+=1，其实区别不大
        return [features, clf]

    with open(pos_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            one_sample = string_to_vector(lex, line, [1, 0])  # [array([ 0.,  1.,  0., ...,  0.,  0.,  0.]), [1,0]]
            dataset.append(one_sample)
    with open(neg_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            one_sample = string_to_vector(lex, line, [0, 1])  # [array([ 0.,  0.,  0., ...,  0.,  0.,  0.]), [0,1]]]
            dataset.append(one_sample)

    # print(len(dataset))
    return dataset


dataset = normalize_dataset(lex)
random.shuffle(dataset)
"""
#把整理好的数据保存到文件，方便使用。到此完成了数据的整理工作
with open('save.pickle', 'wb') as f:
	pickle.dump(dataset, f)
"""

# 取样本中的10%做为测试数据
test_size = int(len(dataset) * 0.1)

dataset = np.array(dataset)

train_dataset = dataset[:-test_size]  #(9596, 2)
test_dataset = dataset[-test_size:]  # (1066, 2)

# Feed-Forward Neural Network
# 定义每个层有多少'神经元''
n_input_layer = len(lex)  # 输入层

n_layer_1 = 1000  # hide layer
n_layer_2 = 1000  # hide layer(隐藏层)听着很神秘，其实就是除输入输出层外的中间层

n_output_layer = 2  # 输出层


# 定义待训练的神经网络
def neural_network(data):
    # 定义第一层"神经元"的权重和biases
    layer_1_w_b = {'w_': tf.Variable(tf.random_normal([n_input_layer, n_layer_1])),
                   'b_': tf.Variable(tf.random_normal([n_layer_1]))}
    # 定义第二层"神经元"的权重和biases
    layer_2_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_1, n_layer_2])),
                   'b_': tf.Variable(tf.random_normal([n_layer_2]))}
    # 定义输出层"神经元"的权重和biases
    layer_output_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_2, n_output_layer])),
                        'b_': tf.Variable(tf.random_normal([n_output_layer]))}

    # w·x+b
    layer_1 = tf.add(tf.matmul(data, layer_1_w_b['w_']), layer_1_w_b['b_'])
    layer_1 = tf.nn.relu(layer_1)  # 激活函数
    layer_2 = tf.add(tf.matmul(layer_1, layer_2_w_b['w_']), layer_2_w_b['b_'])
    layer_2 = tf.nn.relu(layer_2)  # 激活函数
    layer_output = tf.add(tf.matmul(layer_2, layer_output_w_b['w_']), layer_output_w_b['b_'])

    return layer_output


X = tf.placeholder('float', [None, len(train_dataset[0][0])])
# [None, len(train_x)]代表数据数据的高和宽（矩阵），好处是如果数据不符合宽高，tensorflow会报错，不指定也可以。
Y = tf.placeholder('float')



# 使用数据训练神经网络
def train_neural_network0(X, Y):
    predict = neural_network(X)  # 得到预测结果（通过神经网络）
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = predict, labels = Y))  # 定义损失函数
    optimizer = tf.train.AdamOptimizer().minimize(cost_func)  # learning rate 默认 0.001    调用Adam优化算法：是一个寻找全局最优点的优化算法，引入了二次方梯度校正。

    epochs = 13  # 13次整体迭代
    batch_size = 50     # 每次使用50条数据进行训练
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())   #初始化参数，并运行参数初始化方法；
        #初始训练集误差为0
        epoch_loss = 0

        i = 0
        random.shuffle(train_dataset)  # 数据集随机
        train_x = train_dataset[:, 0]  # train_x   (10662,)
        train_y = train_dataset[:, 1]  # 标签
        # 训练过程
        dataSize = len(train_x)
        for epoch in range(epochs):
            while i < dataSize:
                start = i
                end = i + batch_size  # 训练了一个batch大小

                batch_x = train_x[start:end]
                batch_y = train_y[start:end]
                # 运行优化函数
                # 这里返回一个[optimizer,cost]的list, 其中 _代表optimizer,batch_cost代表cost的值
                _, c = session.run([optimizer, cost_func], feed_dict={X: list(batch_x), Y: list(batch_y)})
                epoch_loss += c  # 一个epoch内一个batch的损失
                i += batch_size  # 下一个开始位置
                print(epoch, " *",i)
            print(epoch,' : ', epoch_loss)  # 输出  所有数据后的一次迭代

        text_x = test_dataset[:, 0]  # 测试数据
        text_y = test_dataset[:, 1]  # 标签

        # tf.argmax(vector, 1)：返回的是vector中的最大值的索引号，如果vector是一个向量，那就返回一个值，
        # 如果是一个矩阵，那就返回一个向量，这个向量的每一个维度都是相对应矩阵行的最大值元素的索引号。1代表水平方向
        # tf.equal():返回布尔值，相等返回1，否则0
        # 最后返回大小[none,1]的向量，1所在位置为布尔类型数据
        correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))

        # tf.cast():将布尔型向量转换成浮点型向量
        # tf.reduce_mean():求所有数的均值
        # 返回正确率：也就是所有为1的数目占所有数目的比例
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('准确率: ', accuracy.eval({X: list(text_x), Y: list(text_y)}))

        # print("Train_accuracy :",session.run(accuracy,feed_dict={X:list(text_x), Y: list(text_y)}))

def train_neural_network(X, Y,train_dataset,test_dataset):
    predict = neural_network(X)  # 得到预测结果（通过神经网络）
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = predict, labels = Y))  # 定义损失函数
    optimizer = tf.train.AdamOptimizer().minimize(cost_func)  # learning rate 默认 0.001    调用Adam优化算法：是一个寻找全局最优点的优化算法，引入了二次方梯度校正。

    epochs = 13  # 13次整体迭代
    batch_size = 50     # 每次使用50条数据进行训练

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    #初始训练集误差为0
    epoch_loss = 0
    i = 0
    random.shuffle(train_dataset)  # 数据集随机
    train_x = train_dataset[:, 0]  # train_x   (10662,)
    train_y = train_dataset[:, 1]  # 标签
    # 训练过程
    dataSize = train_x.shape[0]
    for epoch in range(epochs):
        while i < dataSize:
            start = i
            end = i + batch_size  # 训练了一个batch大小

            batch_x = train_x[start:end]
            batch_y = train_y[start:end]
            # 运行优化函数
            # 这里返回一个[optimizer,cost]的list, 其中 _代表optimizer,batch_cost代表cost的值
            _, c = sess.run([optimizer, cost_func], feed_dict={X: list(batch_x), Y: list(batch_y)})
            epoch_loss += c  # 一个epoch内一个batch的损失
            i += batch_size  # 下一个开始位置
            print(epoch, " *",i)
        print(epoch,' : ', epoch_loss)  # 输出  所有数据后的一次迭代

        text_x = test_dataset[:, 0]  # 测试数据
        text_y = test_dataset[:, 1]  # 标签

        # tf.argmax(vector, 1)：返回的是vector中的最大值的索引号，如果vector是一个向量，那就返回一个值，
        # 如果是一个矩阵，那就返回一个向量，这个向量的每一个维度都是相对应矩阵行的最大值元素的索引号。1代表水平方向
        # tf.equal():返回布尔值，相等返回1，否则0
        # 最后返回大小[none,1]的向量，1所在位置为布尔类型数据
    correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))

        # tf.cast():将布尔型向量转换成浮点型向量
        # tf.reduce_mean():求所有数的均值
        # 返回正确率：也就是所有为1的数目占所有数目的比例
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    # print('准确率: ', accuracy.eval({X: list(text_x), Y: list(text_y)}))

    print("Train_accuracy :",sess.run(accuracy,feed_dict={X:list(text_x), Y: list(text_y)}))

train_neural_network(X, Y,train_dataset,test_dataset)

