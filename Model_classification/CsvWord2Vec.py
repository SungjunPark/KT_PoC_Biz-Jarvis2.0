# -*- coding: utf-8 -*-
"""
Created on April ~ September. 2019

        for KT IT Solution Day

@author: SungJun Park, @KT
"""

from konlpy.tag import Okt
from gensim.models import Word2Vec
import csv
import os

os.chdir("..")

twitter = Okt()

file = open("./Data/poc_train_data2.csv", 'r', encoding='UTF-8')
line = csv.reader(file)
token = []
embeddingmodel = []

for i in line:
    content = i[4]  # csv에서 뉴스 제목 또는 뉴스 본문 column으로 변경
    sentence = twitter.pos(i[4], norm=True, stem=True)
    temp = []
    temp_embedding = []
    all_temp = []
    for k in range(len(sentence)):
        temp_embedding.append(sentence[k][0])
        temp.append(sentence[k][0] + '/' + sentence[k][1])
    all_temp.append(temp)
    embeddingmodel.append(temp_embedding)
    category = i[2]  # csv에서 category column으로 변경
    category_number_dic = {'불편접수': 0, '단순문의': 1, '직원칭찬': 2, '지연접수': 3, '해지문의': 4, '기타': 5}
    all_temp.append(category_number_dic.get(category))
    token.append(all_temp)
print("토큰 처리 완료")


embeddingmodel = []
for i in range(len(token)):
    temp_embeddingmodel = []
    for k in range(len(token[i][0])):
        temp_embeddingmodel.append(token[i][0][k])
    embeddingmodel.append(temp_embeddingmodel)
embedding = Word2Vec(embeddingmodel, size=300, window=5, min_count=10, iter=5, sg=1, max_vocab_size=360000000)
embedding.save('./Model_Classification/Word2Vec_model/post.embedding')
