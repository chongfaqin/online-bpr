#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
from kafka import KafkaProducer
from MF_BPR import BPR
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras.optimizers import Adam,SGD

def error_callback(exc):
    raise Exception('Error while sendig data to kafka: {0}'.format(str(exc)))

def write_to_kafka_after_sleep(topic_name, items):
  producer = KafkaProducer(bootstrap_servers=['10.49.201.3:9092'])
  for key,message in items.items():
    val=key+":"+message
    producer.send(topic_name,key=key.encode('utf-8'),value=val.encode('utf-8')).add_errback(error_callback)
  producer.flush()

batch_dict={}
batch_count=100
online_train_count=0
learning_rate = 0.1
epoche=30
item_check_count=512
checkpoint_path = "model_training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
# 创建一个检查点回调
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_weights_only=True,verbose=1,save_freq=epoche)
bpr = BPR()
bpr.summary()
bpr.compile(optimizer=SGD(learning_rate=learning_rate))
bpr.load_weights(checkpoint_path)
if __name__ == "__main__":
    # data = pd.read_csv('data/ratings_small.csv')
    # dp = Data_preprocessor(data)
    # train,test = dp.preprocess()
    online_train_ds = tfio.experimental.streaming.KafkaGroupIODataset(
        topics=["online_training"],
        group_id="bpr",
        servers="10.49.201.3:9092",
        stream_timeout=-1,  # in milliseconds, to block indefinitely, set it to -1.
        configuration=[
            "session.timeout.ms=7000",
            "max.poll.interval.ms=8000",
            "auto.offset.reset=earliest"
        ],
    )

    for (raw_message,key) in online_train_ds:
        online_train_count+=1
        print(online_train_count,raw_message)
        message = tf.io.decode_csv(raw_message, [[0] for i in range(3)])
        #print("1",message)
        #print(message[0].shape)
        message=[tf.reshape(message[0],shape=(1,1)),tf.reshape(message[1],shape=(1,1)),tf.reshape(message[2],shape=(1,1))]
        #print("2",message)
        if(online_train_count%item_check_count==0):
            bpr.fit(message,None,epochs=epoche,callbacks = [cp_callback])
        else:
            bpr.fit(message, None, epochs=epoche)
        # print(message[0].numpy(),bpr.embed_user_layers(message[0])[0][0].numpy())
        user_id=str(message[0].numpy()[0][0])
        p_goods=str(message[1].numpy()[0][0])
        n_goods = str(message[2].numpy()[0][0])
        batch_dict["uv:"+user_id] = ",".join([str(v) for v in np.round(bpr.embed_user_layers(message[0])[0][0],4)])
        batch_dict["gv:"+p_goods] = ",".join([str(v) for v in np.round(bpr.embed_item_layers(message[1])[0][0],4)])
        batch_dict["gv:"+n_goods] = ",".join([str(v) for v in np.round(bpr.embed_item_layers(message[2])[0][0],4)])
        #print(user_id,batch_dict["uv:"+user_id])
        if(len(batch_dict)>=batch_count):
           write_to_kafka_after_sleep("online_param",batch_dict)
           batch_dict.clear()
