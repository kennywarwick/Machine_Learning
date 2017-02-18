#coding:UTF-8
from numpy import *

class ClassNaiveBayes:
    # 創建實驗樣本
    train_doc_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                      ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                      ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him', 'ugly'],
                      ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                      ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                      ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid','ugly']]
    # 傾向性向量, 0:正面 1:負面
    train_doc_sent_vec = [0, 1, 0, 1, 0, 1]

    # 單詞列列表集合
    word_list = []

    # 正面和負面機率, 先驗機率
    p0_v = 0
    p1_v = 0
    p_ab = 0
    # 創建一個包含在所有文檔中出現不會重複詞的列表

    @staticmethod
    def create_word_list(data_list):
        # create empty set
        word_set = set([]) ### 創建一個空集合，set數據類型，返回一個不重複詞表
        for document in data_list:
            # union of the two sets(聯集)(或)
            word_set = word_set | set(document)
        return list(word_set)

    # 將輸入的單詞列表(需要訓練或是測試的)根據上面的單詞集合生成向量
    @staticmethod
    def words_to_vec(word_list, new_word_list):
       vec = [0] * len(word_list) ### 建立N個0的數字陣列
       for word in new_word_list:
           if word in word_list:
               vec[word_list.index(word)] += 1
           else:
               pass
               # print("the word: %s is not in my Vocabulary!" % word)
       return vec

    # Bayes的算法，根據訓練矩陣數據和預先設定的傾向性，來得到 p0，p1 機率。
    @staticmethod
    def train_nb0(train_matrix, train_category):
        num_train_docs = len(train_matrix)
        num_words = len(train_matrix[0])
        p_ab = sum(train_category) / float(num_train_docs)
        # 創建給定長度的填滿1的數組
        p0_num = ones(num_words)
        p1_num = ones(num_words)
        p0_d = 2.0
        p1_d = 2.0
        for i in range(num_train_docs):
            if train_category[i] == 1:
                p1_num += train_matrix[i]
                p1_d += sum(train_matrix[i])
            else:
                p0_num += train_matrix[i]
                p0_d += sum(train_matrix[i])
        p1_v = log(p1_num / p1_d)
        p0_v = log(p0_num / p0_d)
        return p0_v, p1_v, p_ab

    @staticmethod
    def classify_nb(vec, p0_vec, p1_vec, p_class1):
        # element-wise mult
        p0 = sum(vec * p0_vec) + log(1.0 - p_class1)
        p1 = sum(vec * p1_vec) + log(p_class1)
        print('p0:', p0, 'p1:', p1)
        if p1 > p0:
            return 1
        else:
            return 0

    def train(self):
        # 生成單詞列表集合
        self.word_list = self.create_word_list(self.train_doc_list)
        # 訓練矩陣初始化
        train_matrix = []
        # 根劇訓練文檔進行循環
        for post_in_doc in self.train_doc_list:
            # 構建訓練矩陣，將單詞列表轉化為向量
            train_matrix.append(self.words_to_vec(self.word_list, post_in_doc))
        # 根據訓練矩陣和情感分析向量進行訓練，得到
        self.p0_v, self.p1_v, self.p_ab = self.train_nb0(array(train_matrix), array(self.train_doc_sent_vec))
    def testing_nb(self, test_word_list):
        # 對輸入的内容轉化為向量
        this_post_vec = array(self.words_to_vec(self.word_list, test_word_list))
        # 返回分類的值
        return self.classify_nb(this_post_vec, self.p0_v, self.p1_v, self.p_ab)

