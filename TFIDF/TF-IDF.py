

# coding:utf-8
# __author__ = "liuxuejiang"
import jieba
import jieba.posseg as pseg
import os
import sys
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import json

if __name__ == "__main__":
    #     corpus=["我 來到 北京 清華大學", # 第一類文本斷詞後的结果，詞之間以空格隔開
    #         "他 來到 了 網易 杭研 大廈", # 第二類文本的斷詞結果
    #         "小明 碩士 畢業 於 中國 科學院", # 第三類文本的斷詞結果
    #         "我 愛 北京 天安門"] # 第四類文本的斷詞結果
    corpus = []
    with open("E:/AB104/Expedia/Hotels-City-Suites-Kaohsiung-Chenai_comments.json", "r") as a:
        Com_list = json.load(a)
        for i in Com_list:
            for j in i["comment_collection"]:
                corpus.append(j["comment"])

    vectorizer = CountVectorizer(ngram_range=(2,2))  # 該類會將文本中的詞語轉換為詞頻矩陣，矩陣元素a[i][j] 表示j詞在i類文本下的詞頻
    transformer = TfidfTransformer()  # 該類會統計每個詞語的 tf-idf 權重
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))  # 第一個 fit_transform是計算 tf-idf，第二個 fit_transform是將文本轉為詞頻矩陣
    word = vectorizer.get_feature_names()  # 獲取詞代模型中的所有詞語
    weight = tfidf.toarray()  # 將tf-idf矩陣抽取出来，元素a[i][j]表示j詞在i類文本中的tf-idf權重
    for i in range(len(weight)):  # 打印每類文本的tf-idf詞語權重，第一個 for便利所有文本，第二個 for便利某一類文本下的詞語權重
        print u"-------這裡輸出第", i + 1, u"類文本的詞語tf-idf權重------"
        for j in range(len(word)):
            print word[j], weight[i][j]
