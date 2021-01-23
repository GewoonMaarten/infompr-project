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

tokenizer = BertTokenizer.from_pretrained("bert-base-cased") \
    if text_use_bert \
    else RobertaTokenizer.from_pretrained("roberta-base")


def __convert_example_to_feature(txt):
    txt = txt.numpy().decode('utf-8')
    encodings = tokenizer.encode_plus(txt,
                                      add_special_tokens=True,
                                      max_length=text_max_length,
                                      padding='max_length',
                                      truncation=True,
                                      return_attention_mask=True)
    return encodings['input_ids'], encodings['attention_mask']


def __map_example_to_dict(input_ids, attention_masks, label):
    return {"input_ids": input_ids, "attention_mask": attention_masks}, label


def __encode_examples(x, y):
    x = tf.py_function(__convert_example_to_feature,
                       [x], 
                       (tf.uint32, tf.uint32))
    return tf.data.Dataset.from_tensors((x[0], x[1], y))


def __load_base_data(mode):
    try:
        df_path = DF_PATHS[mode]
    except KeyError:
        raise KeyError(
            f'mode can only be "train", "test" or "validate", '
            f'actual value: {mode}')

    df = pd.read_csv(df_path, sep='\t', header=0)
    labels = tf.cast(df['2_way_label'].values, tf.float32)
    return tf.data.Dataset.from_tensor_slices((df['clean_title'].values, labels))


def text_dataset(mode):
    return __load_base_data(mode) \
        .shuffle(buffer_size=50000, reshuffle_each_iteration=True) \
        .interleave(
            __encode_examples,
            num_parallel_calls=tf.data.AUTOTUNE,
            cycle_length=tf.data.AUTOTUNE,
            deterministic=False) \
        .batch(training_batch_size, drop_remainder=True) \
        .map(__map_example_to_dict) \
        .cache() \
        .prefetch(buffer_size=tf.data.AUTOTUNE)
