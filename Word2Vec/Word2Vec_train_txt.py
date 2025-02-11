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
    with open(filename, 'r' ,encoding='utf-8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]   # header 제외 #
    return data

train_data = read_data('ratings_train.txt')
test_data = read_data('ratings_test.txt')

pos_tagger = Okt()

def tokenize(doc):

    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]


## training Word2Vec model using skip-gram
tokens = [tokenize(row[1]) for row in train_data]
model = gensim.models.Word2Vec(size=300 ,sg = 1, alpha=0.025 ,min_alpha=0.025, seed=1234)
model.build_vocab(tokens)

model.train(tokens, epochs=model.iter, total_examples=model.corpus_count)

# for epoch in range(30):

#    model.train(tokens ,model.corpus_count ,epochs = model.iter)
#    model.alpha -= 0.002
#    model.min_alpha = model.alpha

os.chdir("/Users/sungjunpark/POC_BizJarvis_2.0_TA/Word2Vec")
model.save('Movie_review.model')
model.most_similar('팝콘/Noun' ,topn = 20)  ## topn = len(model.wv.vocab)
