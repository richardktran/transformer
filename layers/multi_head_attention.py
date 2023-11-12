import tensorflow as tf


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    :param q: (batch_size, num_heads, sequence_length, d_head)
    :param k: (batch_size, num_heads, sequence_length, d_head)
    :param v: (batch_size, num_heads, sequence_length, d_head)
    :param mask: (batch_size, 1, 1, sequence_length)
    :return: (batch_size, num_heads, sequence_length, d_head), (batch_size, num_heads, sequence_length, sequence_length)
    """
    # (batch_size, num_heads, sequence_length, sequence_length)
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # Scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Add mask to scaled tensor
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # (batch_size, num_heads, sequence_length, sequence_length)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # (batch_size, num_heads, sequence_length, d_head)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_head = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model

        # kwq_layer (batch_size, seq_len, d_model * 3) -> (_, vocab_size, 1536)
        # 8 heads for Q, 8 heads for K, 8 heads for V (_, vocab_size, 512)
        # Each head of Q, K, V has 64 dimensions (512 / 8) -> (_, vocab_size, 64)
        # Matrix QKV has shape (_, vocab_size, 192) -> 64 * 3
        self.kwq_layer = tf.keras.layers.Dense(self.d_model * 3)

    def call(self, x, training=True, mask=None):
        batch_size, sequence_length, d_model = x.shape
        qkv = self.kwq_layer(x)  # (batch_size, sequence_length, d_model * 3)

        # (64, 100, 8, 192) -> This is the shape of Q, K, V of each head
        qkv = tf.reshape(qkv, (batch_size, sequence_length, self.num_heads, self.d_head * 3))

        # Transpose to (64, 8, 100, 192) because we want to split the last dimension into 3 parts
        qkv = tf.transpose(qkv, perm=[0, 2, 1, 3])

        # Split the last dimension into 3 parts (64, 8, 100, 64)
        q, k, v = tf.split(qkv, 3, axis=-1)  # (batch_size, num_heads, sequence_length, d_head)

        # values (batch_size, num_heads, sequence_length, d_head)
        # attention_weights (batch_size, num_heads, sequence_length, sequence_length)
        values, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        # (batch_size, sequence_length, num_heads, d_head)
        # Transpose because we want to concatenate the heads
        values = tf.transpose(values, perm=[0, 2, 1, 3])

        # (batch_size, sequence_length, d_model)
        # Concatenate the heads to get the original shape
        concat_values = tf.reshape(values, (batch_size, sequence_length, self.d_model))

        final_output = tf.keras.layers.Dense(self.d_model)(concat_values)

        return final_output
