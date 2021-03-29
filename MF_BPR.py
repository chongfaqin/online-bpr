import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Input

class BPR(Model):
    def __init__(self):
        """
        BPR
        :param feature_columns: A list. user feature columns + item feature columns
        :param embed_reg: A scalar.  The regularizer of embedding.
        """
        super(BPR, self).__init__()
        # field num
        self.user_field_num = 1
        self.item_field_num = 1
        # user embedding layers [id, age,...]
        self.embed_user_layers = Embedding(4000000, 100)
        # item embedding layers [id, cate_id, ...]
        self.embed_item_layers = Embedding(60000, 100)

    def call(self, inputs):
        # print("inputs:",inputs)
        user_inputs,pos_inputs,neg_inputs=inputs  # (None, user_field_num), (None, item_field_num)
        # user_inputs=tf.keras.layers.Input(shape=(None,1))
        # pos_inputs=tf.keras.layers.Input(shape=(None,1))
        # neg_inputs=tf.keras.layers.Input(shape=(None,1))
        # user info
        user_embed = self.embed_user_layers(tf.squeeze(user_inputs,axis=-1))
        # item  info
        pos_embed = self.embed_item_layers(tf.squeeze(pos_inputs,axis=-1))
        neg_embed = self.embed_item_layers(tf.squeeze(neg_inputs,axis=-1))
        # calculate positive item scores and negative item scores
        pos_scores = tf.reduce_sum(tf.multiply(user_embed, pos_embed), axis=1, keepdims=True)  # (None, 1)
        neg_scores = tf.reduce_sum(tf.multiply(user_embed, neg_embed), axis=1, keepdims=True)  # (None, 1)
        # add loss. Computes softplus: log(exp(features) + 1)
        # self.add_loss(tf.reduce_mean(tf.math.softplus(neg_scores - pos_scores)))
        loss=tf.reduce_mean(-tf.math.log(tf.clip_by_value(tf.nn.sigmoid(pos_scores - neg_scores),1e-8,1.0)))
        self.add_loss(loss)
        return pos_scores,neg_scores

    def summary(self):
        user_inputs = Input(shape=(None,self.user_field_num ), dtype=tf.int32)
        pos_inputs = Input(shape=(None,self.item_field_num), dtype=tf.int32)
        neg_inputs = Input(shape=(None,self.item_field_num), dtype=tf.int32)
        Model(inputs=[user_inputs, pos_inputs, neg_inputs],outputs=self.call([user_inputs, pos_inputs, neg_inputs])).summary()


def test_model():
    user_features = [{'feat': 'user_id', 'feat_num': 100, 'embed_dim': 8}]
    item_features = [{'feat': 'item_id', 'feat_num': 100, 'embed_dim': 8},
                    {'feat': 'cate_id', 'feat_num': 100, 'embed_dim': 8}]
    features = [user_features, item_features]
    model = BPR()
    model.summary()