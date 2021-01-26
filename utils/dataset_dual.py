from transformers import RobertaTokenizer, BertTokenizer
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

def __load_base_data(mode):
    try:
        df_path = DF_PATHS[mode]
    except KeyError:
        raise KeyError(
            f'mode can only be "train", "test" or "validate", '
            f'actual value: {mode}')
    
    df = pd.read_csv(df_path, sep='\t', header=0)

    labels = tf.keras.utils.to_categorical(df['2_way_label'], num_classes=2)
    paths = df['id'].apply(lambda x: dataset_images_path + x + '.jpg')

    return tf.data.Dataset.from_tensor_slices(
        (paths.values, df['clean_title'].values, labels))

def __convert_example_to_feature(txt):
    txt = txt.numpy().decode('utf-8')
    encodings = tokenizer.encode_plus(txt,
                                      add_special_tokens=True,
                                      max_length=text_max_length,
                                      padding='max_length',
                                      truncation=True,
                                      return_attention_mask=True)
    return encodings['input_ids'], encodings['attention_mask']


def __encode_examples(img, title, label):
    x0, x1 = tf.py_function(__convert_example_to_feature,
                            [title],
                            (tf.int32, tf.int32))
    return img, (x0, x1), label


def __load_image(img_path, title, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_height, img_width])

    return tf.data.Dataset.from_tensors((img, title, label))

def __preprocess_image(img, title, label):
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)

    return (img, title), label

def dual_dataset(mode, use_bert):
    global tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased") \
        if use_bert \
        else RobertaTokenizer.from_pretrained("roberta-base")

    return __load_base_data(mode) \
        .shuffle(buffer_size=50000, reshuffle_each_iteration=True) \
        .interleave(
            __load_image, 
            num_parallel_calls=tf.data.AUTOTUNE, 
            cycle_length=tf.data.AUTOTUNE,
            deterministic=False) \
        .map(
            __encode_examples,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False) \
        .batch(training_batch_size, drop_remainder=True) \
        .map(
            __preprocess_image, 
            num_parallel_calls=tf.data.AUTOTUNE, 
            deterministic=False) \
        .prefetch(buffer_size=tf.data.AUTOTUNE)
