import tensorflow as tf
from layers.feed_forward import FeedForwardLayer


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1, epsilon=0.1):
        """
        Encoder Layer contains: Multi-Head Attention, Feed Forward Layer, Layer Normalization.
        :param d_model: (int) dimension of model (embedding dimension)
        :param num_heads: (bool) number of heads
        :param d_ff: (int) dimension of feed forward layer
        :param dropout_rate: (float) dropout rate
        :param epsilon: (float) epsilon value for layer normalization
        """
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardLayer(d_model, d_ff)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=epsilon)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=epsilon)

    def call(self, x, training=True, mask=None):
        """
        :param x: (batch_size, input_seq_len, d_model)
        :param training: (bool)
        :param mask: (batch_size, 1, 1, input_seq_len)
        :return: (batch_size, input_seq_len, d_model)
        """
        x_residual = x.clone()
        x = self.attention(x, x, x, mask)
        x = self.dropout1(x)
        x = self.norm1(x + x_residual)

        x_residual = x.clone()
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + x_residual)

        return x

