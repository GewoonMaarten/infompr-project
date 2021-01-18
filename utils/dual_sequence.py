import pandas as pd
from transformers import RobertaTokenizer

import tensorflow as tf
import pandas as pd

from utils.config import (
    dataset_test_path,
    dataset_train_path,
    dataset_validate_path,
    dataset_images_path,
    img_height,
    img_width,
    training_batch_size,
    text_max_length)


DF_PATHS = {
    'train': dataset_train_path,
    'test': dataset_test_path,
    'validate': dataset_validate_path
}

roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


def convert_example_to_feature(review):
    # combine step for tokenization, WordPiece vector mapping and will
    # add also special tokens and truncate reviews longer than our max length
    return roberta_tokenizer.encode_plus(review,
                                         # add [CLS], [SEP]
                                         add_special_tokens=True,
                                         max_length=text_max_length,  # max length of the text that can go to RoBERTa
                                         # add [PAD] tokens at the end of sentence
                                         pad_to_max_length=True,
                                         return_attention_mask=True,  # add attention mask to not focus on pad tokens
                                         )

class DualDataset(tf.data.Dataset):
    def _generator(mode):
        try:
            df_path = DF_PATHS[mode.decode('utf-8')]
        except KeyError:
            raise KeyError(
                f'mode can only be "train", "test" or "validate", '
                f'actual value: {mode}')

        df = pd.read_csv(df_path, sep='\t', header=0)
        for _, r in df.iterrows():
            img_path = dataset_images_path + f'{r.id}.jpg'
            img = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [img_height, img_width])
            img = tf.cast(img, tf.float32)
            img = tf.keras.applications.efficientnet.preprocess_input(img)

            title = convert_example_to_feature(r.clean_title)['input_ids']

            yield \
                (img, title), \
                tf.cast(r['2_way_label'], tf.float32)

    def __new__(cls, mode):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=(
                (tf.TensorSpec(shape=(img_height, img_width, 3), dtype=tf.float32),
                 tf.TensorSpec(shape=(text_max_length,), dtype=tf.uint32)),
                tf.TensorSpec(shape=(), dtype=tf.float32)),
            args=(tf.constant(mode, dtype=tf.string),)
        )


def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=100)
    ds = ds.batch(training_batch_size, drop_remainder=True)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds
