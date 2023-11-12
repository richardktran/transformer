import tensorflow as tf

from layers.multi_head_attention import MultiHeadAttention


class TestMultiHeadAttention(tf.test.TestCase):
    def test_call(self):
        d_model = 64
        num_heads = 4
        sequence_length = 10
        batch_size = 16

        # Create an instance of MultiHeadAttention
        multi_head_attention = MultiHeadAttention(d_model, num_heads)

        # Generate some random input data
        input_data = tf.random.normal((batch_size, sequence_length, d_model))

        # Call the MultiHeadAttention layer
        output = multi_head_attention(input_data)

        # Check the output shape
        expected_output_shape = (batch_size, sequence_length, d_model)
        self.assertEqual(output.shape, expected_output_shape)
