#!/usr/bin/env python
# coding: utf-8

import os
from datetime import datetime
import time
import threading
import json
import numpy as np
from kafka import KafkaProducer
from kafka.errors import KafkaError
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio
import random

train_sample_size =1

def error_callback(exc):
    raise Exception('Error while sendig data to kafka: {0}'.format(str(exc)))

def write_to_kafka_after_sleep(topic_name, items):
  time.sleep(30)
  print("#"*100)
  print("Writing messages into topic: {0} after a nice sleep !".format(topic_name))
  print("#"*100)
  count=0
  producer = KafkaProducer(bootstrap_servers=['10.49.201.3:9092'])
  for message, key in items:
    producer.send(topic_name,key=key.encode('utf-8'),value=message.encode('utf-8')).add_errback(error_callback)
    count+=1
  producer.flush()
  print("#"*100)
  print("Wrote {0} messages into topic: {1}".format(count, topic_name))
  print("#"*100)

if __name__=='__main__':
    data = pd.read_csv('data/ratings_small.csv')
    print(data)

    # x_train = list(filter(None, data.to_csv(index=False).split("\n")[1:]))
    # print("x_train:",x_train)

    all_item = set(data['movieId'].unique())

    #count
    num_user = len(data['userId'].unique())
    num_item = len(data['movieId'].unique())
    num_event = len(data)
    print("all:",num_event,"user_count:",num_user,"item_count:",num_item)

    #code
    user_id = data['userId'].unique()
    user_id_map = {user_id[i]: i for i in range(num_user)}
    item_id = data['movieId'].unique()
    item_id_map = {item_id[i]: i for i in range(num_item)}
    # print("user_id_map:",user_id_map)
    # print("item_id_map:",item_id_map)


    #construct train data
    training_data = data.loc[:, ['userId', 'movieId']].values
    training_data = [[user_id_map[training_data[i][0]], item_id_map[training_data[i][1]]] for i in range(num_event)]

    #negative sample
    user_session = data.groupby('userId')['movieId'].apply(set).reset_index().loc[:,['movieId']].values.reshape(-1)
    for td in training_data:
        td.extend([item_id_map[s] for s in random.sample(all_item.difference(user_session[td[0]]),train_sample_size)])
    # print(training_data)

    trainng_df=pd.DataFrame(training_data,columns=["userIdx","pMovieIdx","nMovieIdx"])
    x_train = list(filter(None, trainng_df.to_csv(index=False).split("\n")[1:]))
    key_train = list(filter(None, data["timestamp"].to_csv(index=False).split("\n")[1:]))
    print(x_train)

    write_to_kafka_after_sleep("online_training", zip(x_train,key_train))
    # thread = threading.Thread(target=write_to_kafka_after_sleep,args=("online_training", zip(key_train,x_train)))
    # thread.daemon = True
    # thread.start()