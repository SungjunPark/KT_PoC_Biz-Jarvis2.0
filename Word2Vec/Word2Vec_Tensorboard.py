# -*- coding: utf-8 -*-
"""
Created on April ~ September. 2019

        for KT IT Solution Day

@author: SungJun Park, @KT
"""

import gensim
import tensorflow as tf
import codecs
import os
import numpy as np
import warnings

#warnings.filterwarnings('ignore')
os.chdir("/Users/sungjunpark/KT_PoC_Biz-Jarvis2.0/Word2Vec")

## import model
model = gensim.models.word2vec.Word2Vec.load('KT100_CallCenter.model')
#model.most_similar('개통/Noun',topn = 3)  ## topn = len(model.wv.vocab)

max_size = len(model.wv.vocab)-1
w2v = np.zeros((max_size,model.layer1_size))


with codecs.open("metadata.tsv", 'w+', encoding='utf8') as file_metadata:
    for i,word in enumerate(model.wv.index2word[:max_size]):
        w2v[i] = model.wv[word]
        file_metadata.write(word + "\n")

from tensorflow.contrib.tensorboard.plugins import projector

sess = tf.InteractiveSession()
##  Create 2D tensor called embedding that holds our embeddings ##
with tf.device("/cpu:0"):
    embedding = tf.Variable(w2v, trainable = False,  name = 'embedding')

tf.global_variables_initializer().run()

path = 'tensorboard_w2v'

saver = tf.train.Saver()
writer = tf.summary.FileWriter(path, sess.graph)

## adding into project
config = projector.ProjectorConfig()
embed = config.embeddings.add()
embed.tensor_name = 'embedding'
embed.metadata_path = '/Users/sungjunpark/KT_PoC_Biz-Jarvis2.0/Word2Vec/metadata.tsv'
# Specify the width and height of a single thumbnail.
projector.visualize_embeddings(writer, config)
saver.save(sess, path + '/model.ckpt' , global_step=max_size)

## cmd 실행후 -> 1. cd /Users/sungjunpark/KT_PoC_Biz-Jarvis2.0/Word2Vec
##              2. tensorboard --logdir=./tensorboard_w2v 입력
