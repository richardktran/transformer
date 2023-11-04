import tensorflow as tf
from nmt_dataset import NMTDataset


if __name__ == '__main__':
    buffer_size = 10000
    batch_size = 64
    nmt_dataset = NMTDataset('en', 'vi', example_length=100)
    train_dataset, val_dataset = nmt_dataset.build_dataset(buffer_size, batch_size, max_sequence_length=100)
    input_vocab_size = nmt_dataset.input_vocab_size
    target_vocab_size = nmt_dataset.target_vocab_size

