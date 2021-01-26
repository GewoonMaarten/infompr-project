import tensorflow as tf
from utils.config import training_batch_size

def configure_for_performance(ds):
    return ds.shuffle(buffer_size=10000, reshuffle_each_iteration=True) \
        .batch(training_batch_size, drop_remainder=True) \
        .cache() \
        .prefetch(buffer_size=tf.data.AUTOTUNE)
