# -*- coding: utf-8 -*-
"""
Created on April ~ September. 2019

        for KT IT Solution Day

@author: SungJun Park, @KT
"""

import os
import tensorflow as tf
import Bi_LSTM
import Word2Vec
import gensim
import numpy as np

W2V = Word2Vec.Word2Vec()

Batch_size = 1
Vector_size = 300
Maxseq_length = 95  ## Max length of training data
learning_rate = 0.001
lstm_units = 128
num_class = 3
keep_prob = 1.0

X = tf.placeholder(tf.float32, shape=[None, Maxseq_length, Vector_size], name='X')
Y = tf.placeholder(tf.float32, shape=[None, num_class], name='Y')
seq_len = tf.placeholder(tf.int32, shape=[None])

BiLSTM = Bi_LSTM.Bi_LSTM(lstm_units, num_class, keep_prob)

with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
    logits = BiLSTM.logits(X, BiLSTM.W, BiLSTM.b, seq_len)
    loss, optimizer = BiLSTM.model_build(logits, Y, learning_rate)

prediction = tf.nn.softmax(logits)


def Convert2Vec(model_name, sentence):
    word_vec = []
    sub = []
    model = gensim.models.word2vec.Word2Vec.load(model_name)
    for word in sentence:
        if (word in model.wv.vocab):
            sub.append(model.wv[word])
        else:
            sub.append(np.random.uniform(-0.25, 0.25, 300))  ## used for OOV words
    word_vec.append(sub)
    return word_vec


saver = tf.train.Saver()
init = tf.global_variables_initializer()
modelName = "./Bi_LSTM_model/BiLSTM_CallCenter_model.ckpt"

sess = tf.Session()
sess.run(init)
saver.restore(sess, modelName)

os.chdir("..")


def Grade(sentence):
    tokens = W2V.tokenize(sentence)

    embedding = Convert2Vec('./Word2Vec/KT100_CallCenter.model', tokens)
    zero_pad = W2V.Zero_padding(embedding, Batch_size, Maxseq_length, Vector_size)
    global sess
    result = sess.run(tf.argmax(prediction, 1), feed_dict={X: zero_pad, seq_len: [len(tokens)]})
    if (result == 1):
        print("긍정입니다")
    elif (result == 2):
        print("중립입니다")
    else:
        print("부정입니다")


while (1):
    s = input("문장을 입력하세요 : ")
    if (s == str(1)):
        break
    else:
        Grade(s)

