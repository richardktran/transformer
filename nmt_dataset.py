import tensorflow as tf
from sklearn.model_selection import train_test_split


def tokenize(lang, max_sequence_length):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer.fit_on_texts(lang)

    vocab_size = len(tokenizer.word_index) + 1

    tensor = tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen=max_sequence_length, padding='post')

    return tensor, vocab_size


class NMTDataset:
    def __init__(self, input_lang, target_lang, example_length=10):
        self.input_lang = input_lang
        self.target_lang = target_lang
        self.example_length = example_length
        self.train_dir = 'data/train'
        self.input_vocab_size = None
        self.target_vocab_size = None

    def read_from_file(self):
        with open('{}/{}.txt'.format(self.train_dir, self.input_lang), 'r') as input_file:
            input_lines = input_file.readlines()[:self.example_length]
        with open('{}/{}.txt'.format(self.train_dir, self.target_lang), 'r') as target_file:
            target_lines = target_file.readlines()[:self.example_length]

        preprocess_text = [(input_text.strip(), target_text.strip()) for input_text, target_text in
                           zip(input_lines, target_lines)]

        return zip(*preprocess_text)

    def load_dataset(self, max_sequence_length):
        input_sentences, target_sentences = self.read_from_file()
        # Tokenize the input and target sentences
        input_tensor, self.input_vocab_size = tokenize(input_sentences, max_sequence_length)
        target_tensor, self.target_vocab_size = tokenize(target_sentences, max_sequence_length)

        return input_tensor, target_tensor

    def build_dataset(self, buffer_size, batch_size, max_sequence_length=100):
        input_tensor, target_tensor = self.load_dataset(max_sequence_length)
        input_train, input_val, target_train, target_val = train_test_split(input_tensor, target_tensor, test_size=0.2)
        train_dataset = tf.data.Dataset.from_tensor_slices((input_train, target_train))
        train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices((input_val, target_val))
        val_dataset = val_dataset.shuffle(buffer_size).batch(batch_size)

        return train_dataset, val_dataset

