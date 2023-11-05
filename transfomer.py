import tensorflow as tf
from layers.encoder import Encoder


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, d_ff, num_heads, input_vocab_size, target_vocab_size, dropout_rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, d_ff, num_heads, input_vocab_size, d_ff, dropout_rate)
        # self.decoder = Decoder(d_model, num_heads, num_layers, target_vocab_size)
        self.feed_forward_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self,
             input_data,
             training=True,
             encoder_mask=None,
             decoder_mask=None,
             cross_mask=None):
        encoder_out = self.encoder(input_data, training, encoder_mask)
        # decoder_out = self.decoder(target_data, encoder_out, training, decoder_mask, cross_mask)
        return self.feed_forward_layer(encoder_out)
