#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
import time
import tensorflow as tf
import random
import math


class BPR():
    '''
    parameter
    train_sample_size : 訓練時，每個正樣本，我sample多少負樣本
    test_sample_size : 測試時，每個正樣本，我sample多少負樣本
    num_k : item embedding的維度大小
    evaluation_at : recall@多少，及正樣本要排前幾名，我們才視為推薦正確
    '''
    def __init__(self,batch_size=32,train_sample_size=10,test_sample_size=50,num_k=100,evaluation_at=10):
        self.batch_size = batch_size
        self.train_sample_size = train_sample_size
        self.test_sample_size = test_sample_size
        self.num_k = num_k
        self.evaluation_at = evaluation_at

        self.sess = tf.Session() #create session
        self.sess.run(tf.global_variables_initializer())

    def build_model(self):
        self.X_user = tf.placeholder(tf.int32,shape=(None , 1))
        self.X_pos_item = tf.placeholder(tf.int32,shape=(None , 1))
        self.X_neg_item = tf.placeholder(tf.int32,shape=(None , 1))

        user_embedding = tf.Variable(tf.truncated_normal(shape=[2000000,self.num_k],mean=0.0,stddev=0.5))
        item_embedding = tf.Variable(tf.truncated_normal(shape=[40000,self.num_k],mean=0.0,stddev=0.5))

        embed_user = tf.nn.embedding_lookup(user_embedding , self.X_user)
        embed_pos_item = tf.nn.embedding_lookup(item_embedding , self.X_pos_item)
        embed_neg_item = tf.nn.embedding_lookup(item_embedding , self.X_neg_item)

        pos_score = tf.matmul(embed_user, embed_pos_item, transpose_b=True)
        neg_score = tf.matmul(embed_user, embed_neg_item, transpose_b=True)

        self.loss = tf.reduce_mean(-tf.log(tf.nn.sigmoid(pos_score-neg_score)))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

    def fit(self,user_id,pos_item_id,neg_item_id,enpochs=3):
        for epoch in range(enpochs):
            _, loss = self.sess.run([self.optimizer, self.loss],feed_dict={self.X_user: user_id, self.X_pos_item:pos_item_id, self.X_neg_item: neg_item_id})
            print("epoch:%d,loss:%.2f"%(epoch, loss))