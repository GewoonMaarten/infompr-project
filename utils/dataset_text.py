import pandas as pd
from transformers import RobertaTokenizer, BertTokenizer

import tensorflow as tf
import pandas as pd

from utils.config import (
    dataset_test_path,
    dataset_train_path,
    dataset_validate_path,
    training_batch_size,
    text_max_length)


DF_PATHS = {
    'train': dataset_train_path,
    'test': dataset_test_path,
    'validate': dataset_validate_path
}


class TextDataset():
    def __init__(self, mode, is_bert) -> None:
        self.mode = mode
        if is_bert:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        else:
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def __convert_example_to_feature(self, txt):
        txt = txt.numpy().decode('utf-8')
        encodings = self.tokenizer.encode_plus(txt,
                                               add_special_tokens=True,
                                               max_length=text_max_length,
                                               padding='max_length',
                                               truncation=True,
                                               return_attention_mask=True)
        return encodings['input_ids'], encodings['attention_mask']

    def __encode_examples(self, x, y):
        x0, x1 = tf.py_function(self.__convert_example_to_feature,
                                [x],
                                (tf.int32, tf.int32))
        return (x0, x1), y

    def __load_base_data(self):
        try:
            df_path = DF_PATHS[self.mode]
        except KeyError:
            raise KeyError(
                f'mode can only be "train", "test" or "validate", '
                f'actual value: {self.mode}')

        df = pd.read_csv(df_path, sep='\t', header=0)
        labels = tf.keras.utils.to_categorical(
            df['2_way_label'], num_classes=2)
        return tf.data.Dataset.from_tensor_slices((df['clean_title'].values, labels))

    def text_dataset(self):
        return self.__load_base_data() \
            .shuffle(buffer_size=50000, reshuffle_each_iteration=True) \
            .map(
                self.__encode_examples,
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=False) \
            .batch(training_batch_size, drop_remainder=True) \
            .prefetch(buffer_size=tf.data.AUTOTUNE)
