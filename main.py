import tensorflow as tf
from nmt_dataset import NMTDataset
from transfomer import Transformer

if __name__ == '__main__':
    buffer_size = 10000
    batch_size = 64
    d_ff = 2048
    dropout_rate = 0.1

    nmt_dataset = NMTDataset('en', 'vi', example_length=2000)

    # train_dataset: (num_sentences x 64 x 100)
    train_dataset, val_dataset = nmt_dataset.build_dataset(buffer_size, batch_size, max_sequence_length=100)

    transformer = Transformer(
        num_layers=6,
        d_model=512,
        d_ff=2048,
        num_heads=8,
        input_vocab_size=nmt_dataset.input_vocab_size,
        target_vocab_size=nmt_dataset.target_vocab_size,
        dropout_rate=dropout_rate
    )

    (en, vi) = next(iter(train_dataset))

    # transformer.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    transformer(en, training=True)

    print(transformer.summary())


    # for (input, target) in train_dataset:
    #     print(input.shape)
    #     print(target.shape)
    #     break

