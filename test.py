import tensorflow as tf
print(tf.__version__)
user_embed=tf.constant([1,1,1])
pos_embed=tf.constant([2,2,2])
pos_scores = tf.reduce_sum(tf.multiply(user_embed, pos_embed), axis=1, keepdims=True)  # (None, 1)
# neg_scores = tf.reduce_sum(tf.multiply(user_embed, neg_embed), axis=1, keepdims=True)  # (None, 1)
tf.print(pos_scores)