import tensorflow as tf
import pandas as pd

from utils.config import (
    dataset_test_path,
    dataset_train_path,
    dataset_validate_path,
    dataset_images_path,
    img_height,
    img_width,
    training_batch_size)

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

    return tf.data.Dataset.from_tensor_slices((paths.values, labels))

def __load_image(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_height, img_width])

    return tf.data.Dataset.from_tensors((img, label))

def __preprocess_image(img, label):
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)

    return img, label

def image_dataset(mode):
    return __load_base_data(mode) \
        .interleave(
            __load_image, 
            num_parallel_calls=tf.data.AUTOTUNE, 
            cycle_length=tf.data.AUTOTUNE,
            deterministic=True) \
        .batch(training_batch_size, drop_remainder=True) \
        .map(
            __preprocess_image, 
            num_parallel_calls=tf.data.AUTOTUNE, 
            deterministic=True) \
        .prefetch(buffer_size=tf.data.AUTOTUNE)
