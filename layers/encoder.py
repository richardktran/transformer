import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Embedding
from layers.encoder_layer import EncoderLayer
from layers.utils import create_positional_encoding


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, d_ff, num_heads, input_vocab_size, dropout_rate=0.1, epsilon=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.input_vocab_size = input_vocab_size
        self.embedding = Embedding(self.input_vocab_size, self.d_model)
        self.encoder_layers = [EncoderLayer(self.d_model, self.num_heads, d_ff, dropout_rate, epsilon) for _ in range(self.num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=True, mask=None):
        """
        :param inputs: (batch_size, input_seq_len)
        :param training: (bool)
        :param mask: (batch_size, 1, 1, input_seq_len)
        :return: (batch_size, input_seq_len, d_model)
        """
        encoder_out = self.embedding(inputs)
        positional_encoding = create_positional_encoding(inputs.shape[1], self.d_model)

        encoder_out = self.dropout(encoder_out + positional_encoding, training=training)

        for encoder_layer in self.encoder_layers:
            encoder_out = encoder_layer(encoder_out, training, mask)

        return encoder_out
