#!/usr/bin/env python
# coding: utf-8
import os
from MF_BPR import BPR
from kafka import KafkaProducer
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras.optimizers import Adam,SGD,Ftrl
import numpy as np
import redis

offline_pool=redis.ConnectionPool(host='ai-offline.redis.rds.inagora.org',port=6379,password='2nHUmr9mZmguwfwb',db=0)
offline_r = redis.StrictRedis(connection_pool=offline_pool,decode_responses=True)

online_pool=redis.ConnectionPool(host='ai-online.redis.rds.inagora.org',port=6379,password='6JGowG87sKPA!',db=0)
online_r = redis.StrictRedis(connection_pool=online_pool,decode_responses=True)

def write_to_offline_redis(items):
    pip = offline_r.pipeline()
    for key, message in items.items():
        # val = key + ":" + message
        pip.set(key,message,ex=24 * 60 * 60 * 90)
    pip.execute()

def write_to_online_redis(items):
    pip=online_r.pipeline()
    for key,message in items.items():
        # print(key,message)
        pip.set(key,message,ex=24 * 60 * 60 * 90)
    pip.execute()

def error_callback(exc):
    raise Exception('Error while sendig data to kafka: {0}'.format(str(exc)))

def write_to_kafka(topic_name, items):
  producer = KafkaProducer(bootstrap_servers=['10.49.201.3:9092','10.49.201.2:9092','10.49.201.4:9092'])
  for key,message in items.items():
    val=key+":"+message
    producer.send(topic_name,key=key.encode('utf-8'),value=val.encode('utf-8')).add_errback(error_callback)
  producer.flush()

batch=1
batch_dict={}
user_batch_dict={}
batch_count=1280
online_train_count=0
learning_rate = 0.01
epoche=1
item_check_count=1024
topic_name="online_param"
kafka_server="10.49.201.3:9092,10.49.201.4:9092,10.49.201.2:9092"

checkpoint_path = "model_training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
# 创建一个检查点回调
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_weights_only=True,verbose=1,save_freq=epoche)
bpr = BPR()
bpr.summary()
bpr.compile(optimizer=SGD(learning_rate=learning_rate))
# bpr.load_weights(checkpoint_path)
if __name__ == "__main__":
    online_train_ds = tfio.experimental.streaming.KafkaBatchIODataset(
        topics=["online_training"],
        group_id="bpr-1202",
        servers=kafka_server,
        stream_timeout=-1,  # in milliseconds, to block indefinitely, set it to -1.
        message_poll_timeout=10000,
        configuration=[
            "session.timeout.ms=7000",
            "max.poll.interval.ms=8000",
            "auto.offset.reset=earliest"
        ],
    )

    for mini_ds in online_train_ds:
        # print("0", mini_ds, type(mini_ds))
        mini_ds = mini_ds.map(lambda m, k: (tf.io.decode_csv(m, [[0] for i in range(3)])))
        # print("1", mini_ds, type(mini_ds))
        mini_ds=mini_ds.batch(batch)
        # print("2", mini_ds, type(mini_ds))
        for mini in mini_ds:
            # print("3",mini,type(mini))
            # bpr.fit(mini, None, epochs=3)
            online_train_count += 1
            if (online_train_count % item_check_count == 0):
                bpr.fit(mini, None, epochs=epoche,steps_per_epoch=1,callbacks=[cp_callback])
            else:
                bpr.fit(mini, None, epochs=epoche,steps_per_epoch=1)

            for item in zip(mini[0].numpy(),bpr.embed_user_layers(mini[0])):
                user_batch_dict["uv:" + str(item[0])]=",".join([str(v) for v in np.round(item[1],4)])
                # batch_dict["uv:" + str(item[0])]=",".join([str(v) for v in np.round(item[1],4)])
                # print("uv:" + str(item[0]), batch_dict["uv:" + str(item[0])])
            for item in zip(mini[1].numpy(),bpr.embed_item_layers(mini[1])):
                batch_dict["gv:" + str(item[0])] = ",".join([str(v) for v in np.round(item[1], 4)])
            for item in zip(mini[2].numpy(),bpr.embed_item_layers(mini[2])):
                batch_dict["gv:" + str(item[0])] = ",".join([str(v) for v in np.round(item[1], 4)])

        if (len(batch_dict) >= batch_count):
            write_to_kafka(topic_name, batch_dict)
            write_to_offline_redis(batch_dict)
            write_to_online_redis(user_batch_dict)
            print(online_train_count, "--------------flush-------------", len(user_batch_dict), len(batch_dict))
            batch_dict.clear()
            user_batch_dict.clear()