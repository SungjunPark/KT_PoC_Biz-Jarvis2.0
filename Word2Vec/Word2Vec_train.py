# -*- coding: utf-8 -*-
"""
Created on April ~ September. 2019.

        for KT IT Solution Day

@author: SungJun Park, @KT
"""

import os
from konlpy.tag import Okt
import gensim
import tensorflow as tf
import numpy as np
import codecs

os.chdir("/Users/sungjunpark/POC_BizJarvis_2.0_TA/Data")


def read_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = [line.split(',') for line in f.read().splitlines()]
        data = data[1:]  # header 제외 #
    return data


train_data = read_data('poc_train_data2.csv')
test_data = read_data('poc_test_data.csv')

pos_tagger = Okt()


def tokenize(doc):
    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]


## training Word2Vec model using skip-gram
tokens = [tokenize(row[4]) for row in train_data]  # csv의 단어 임베딩 할 열 입력 ex) 6
model = gensim.models.Word2Vec(size=300, sg=1, alpha=0.025, min_alpha=0.025, seed=1234, iter=10)
model.build_vocab(tokens)

model.train(tokens, epochs=model.iter, total_examples=model.corpus_count)

# for epoch in range(30):

# model.train(tokens,model.corpus_count,epochs = model.iter)
# model.alpha -= 0.002
# model.min_alpha = model. wnalpha


os.chdir("/Users/sungjunpark/POC_BizJarvis_2.0_TA/Word2Vec")
model.save('KT100_CallCenter.model')
# model.most_similar('가입/Noun', topn = 3)  ## topn = len(model.wv.vocab)
# print(model.most_similar('가입/Noun', topn = 10))
