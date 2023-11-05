import tensorflow as tf


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1, epsilon=0.1):
        super(EncoderLayer, self).__init__()
