#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf


class BPR():
    '''
    parameter
    train_sample_size : 訓練時，每個正樣本，我sample多少負樣本
    test_sample_size : 測試時，每個正樣本，我sample多少負樣本
    num_k : item embedding的維度大小
    evaluation_at : recall@多少，及正樣本要排前幾名，我們才視為推薦正確
    '''
    def __init__(self,n_epochs=1,batch_size=32,train_sample_size=10,test_sample_size=50,num_k=100,evaluation_at=10):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.train_sample_size = train_sample_size
        self.test_sample_size = test_sample_size
        self.num_k = num_k
        self.evaluation_at = evaluation_at

        self.build_model() #build TF graph
        self.sess = tf.Session() #create session
        self.sess.run(tf.global_variables_initializer())
        self.epoch = 0
        self.total_loss = 0

    def build_model(self):
        self.X_user = tf.keras.Input(shape=(None , 1),name="user")
        self.X_pos_item = tf.keras.Input(shape=(None , 1),name="p_item")
        self.X_neg_item = tf.keras.Input(shape=(None , 1),name="n_item")

        # user_embedding = tf.Variable(tf.truncated_normal(shape=[2000000,self.num_k],mean=0.0,stddev=0.5))
        # item_embedding = tf.Variable(tf.truncated_normal(shape=[40000,self.num_k],mean=0.0,stddev=0.5))
        user_embedding = tf.keras.layers.Embedding(2000000,self.num_k)
        item_embedding = tf.keras.layers.Embedding(40000,self.num_k)

        embed_user = user_embedding(self.X_user)
        embed_pos_item = item_embedding(self.X_pos_item)
        embed_neg_item = item_embedding(self.X_neg_item)

        pos_score = tf.matmul(embed_user, embed_pos_item, transpose_b=True)
        neg_score = tf.matmul(embed_user, embed_neg_item, transpose_b=True)

        self.loss = tf.reduce_mean(-tf.math.log(tf.nn.sigmoid(pos_score-neg_score)))
        # self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.001).minimize(self.loss,var_list=[embed_user,embed_pos_item,embed_neg_item])


    def fit(self,user,p_item,n_item):
        # 构建模型
        model = tf.keras.Model(inputs=[user, p_item, n_item],outputs=[self.loss])
        model.summary()
        # loss=tf.keras.optimizers.SGD(learning_rate=0.001).minimize(self.loss,var_list={self.X_user: user, self.X_pos_item:p_item, self.X_neg_item: n_item})
        # self.total_loss += loss
        self.epoch+=1
        print("epoch:%d,loss:%.2f"%(self.epoch, self.total_loss))
