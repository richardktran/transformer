import tensorflow as tf
from layers.utils import create_positional_encoding

max_sequence_length = 100
d_model = 512


class UtilsTest(tf.test.TestCase):
    def test_create_positional_encoding(self):
        positional_encoding = tf.cast(create_positional_encoding(max_sequence_length, d_model), dtype=tf.float32)
        self.assertEqual(positional_encoding.shape, (max_sequence_length, d_model))
        self.assertEqual(positional_encoding.dtype, tf.float32)
