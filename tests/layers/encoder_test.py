import tensorflow as tf
from layers.encoder import Encoder

class EncoderTest(tf.test.TestCase):
    def test_encoder_call(self):
        batch_size = 64
        input_seq_len = 100 # max sequence length
        encoder = Encoder(num_layers=6, d_model=512, d_ff=2048, num_heads=8, input_vocab_size=8500)
        input_data = tf.random.uniform((batch_size, input_seq_len))
        encoder_out = encoder(input_data, training=True)
        self.assertEqual(encoder_out.shape, (batch_size, input_seq_len, 512))

