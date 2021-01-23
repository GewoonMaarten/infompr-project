import pandas as pd
from transformers import RobertaTokenizer, BertTokenizer

import tensorflow as tf
import pandas as pd

from utils.config import (
    dataset_test_path,
    dataset_train_path,
    dataset_validate_path,
    training_batch_size,
    text_max_length,
    text_use_bert)


DF_PATHS = {
    'train': dataset_train_path,
    'test': dataset_test_path,
    'validate': dataset_validate_path
}

tokenizer = BertTokenizer.from_pretrained("bert-base-cased") if text_use_bert else RobertaTokenizer.from_pretrained("roberta-base") 


def convert_example_to_feature(review):
    # combine step for tokenization, WordPiece vector mapping and will
    # add also special tokens and truncate reviews longer than our max length
    return tokenizer.encode_plus(review,
                                         # add [CLS], [SEP]
                                         add_special_tokens=True,
                                         max_length=text_max_length,  # max length of the text that can go to RoBERTa
                                         # add [PAD] tokens at the end of sentence
                                         pad_to_max_length=True,
                                         return_attention_mask=True,  # add attention mask to not focus on pad tokens
                                         )


class DatasetText(tf.data.Dataset):
    def _generator(mode):
        try:
            df_path = DF_PATHS[mode.decode('utf-8')]
        except KeyError:
            raise KeyError(
                f'mode can only be "train", "test" or "validate", '
                f'actual value: {mode}')

        df = pd.read_csv(df_path, sep='\t', header=0)
        for _, r in df.iterrows():
            title = convert_example_to_feature(r.clean_title)['input_ids']

            yield \
                title, \
                tf.cast(r['2_way_label'], tf.float32)

    def __new__(cls, mode):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=(
                tf.TensorSpec(shape=(text_max_length,), dtype=tf.uint32),
                tf.TensorSpec(shape=(), dtype=tf.float32)),
            args=(tf.constant(mode, dtype=tf.string),)
        )
