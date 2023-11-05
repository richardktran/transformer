import numpy as np
import tensorflow as tf


def create_positional_encoding(max_sequence_length, d_model):
    """
    :param max_sequence_length: (int)
    :param d_model: (int)
    :return: (max_sequence_length, d_model)
    """

    even_i = np.arange(0, d_model, 2)
    denominator = np.power(10000, (2 * even_i) / np.float32(d_model))

    positional_encoding = np.zeros((max_sequence_length, d_model))

    positional_encoding[:, 0::2] = np.sin(np.arange(max_sequence_length).reshape(-1, 1) / denominator)
    positional_encoding[:, 1::2] = np.cos(np.arange(max_sequence_length).reshape(-1, 1) / denominator)

    return tf.cast(positional_encoding, dtype=tf.float32)


def visualize_positional_encoding():
    import matplotlib.pyplot as plt
    pos_encoding = create_positional_encoding(50, 512)
    plt.pcolormesh(pos_encoding, cmap='RdBu')
    plt.xlabel('d')
    plt.xlim((0, 512))
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()


# if __name__ == '__main__':
#     visualize_positional_encoding()

