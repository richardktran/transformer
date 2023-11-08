import tensorflow as tf


class FeedForwardLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])

    def call(self, x, training=True, mask=None):
        """
        :param x: (batch_size, input_seq_len, d_model)
        :param training: (bool)
        :param mask: (batch_size, 1, 1, input_seq_len)
        :return: (batch_size, input_seq_len, d_model)
        """
        return self.seq(x)
