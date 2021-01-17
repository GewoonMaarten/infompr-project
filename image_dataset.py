import tensorflow as tf
import pandas as pd
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as inception_resnet_preprocess_input

from utils.image_model_factory import ModelBuilder, ModelType
from utils.config import (
    dataset_test_path,
    dataset_train_path,
    dataset_validate_path,
    dataset_images_path,
    img_height,
    img_width,
    training_batch_size,
    training_epochs)


DF_PATHS = {
    'train': dataset_train_path,
    'test': dataset_test_path,
    'validate': dataset_validate_path
}

PREPROCESSORS = {
    'inceptionv3': inception_preprocess_input,
    'inceptionresnet': inception_resnet_preprocess_input,
    'efficientnet': efficientnet_preprocess_input
}


class Fakeddit(tf.data.Dataset):
    def _generator(mode, model):
        try:
            df_path = DF_PATHS[mode.decode('utf-8')]
        except KeyError:
            raise KeyError(
                f'mode can only be "train", "test" or "validate", '
                f'actual value: {mode}')
        
        try:
            preprocess_input = PREPROCESSORS[model.decode('utf-8')]
        except KeyError:
            raise KeyError(
                f'model can only be "inceptionv3", "inceptionresnet" or '
                f'"efficientnet", actual value: {mode}')

        df = pd.read_csv(df_path, sep='\t', header=0)
        for _, r in df.iterrows():
            img_path = dataset_images_path + f'{r.id}.jpg'
            img = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [img_height, img_width])
            img = tf.cast(img, tf.float32)
            img = preprocess_input(img)

            yield \
                img, \
                tf.keras.utils.to_categorical(r['2_way_label'], num_classes=2)

    def __new__(cls, mode, model):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=(
                tf.TensorSpec(shape=(img_height, img_width, 3),
                              dtype=tf.float32),
                tf.TensorSpec(shape=(2,), dtype=tf.float32)),
            args=(
                tf.constant(mode, dtype=tf.string),
                tf.constant(model, dtype=tf.string))
        )


def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=100)
    ds = ds.batch(training_batch_size, drop_remainder=True)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


if __name__ == "__main__":
    train_ds = configure_for_performance(Fakeddit('train', 'inceptionv3'))
    test_ds = configure_for_performance(Fakeddit('test', 'inceptionv3'))
    validate_ds = configure_for_performance(Fakeddit('validate', 'inceptionv3'))

    model_type = ModelType.INCEPTION

    model_builder = ModelBuilder(model_type, 2)
    model_builder.compile_for_transfer_learning()

    history = model_builder.model.fit(train_ds, validation_data=validate_ds, epochs=training_epochs, batch_size=training_batch_size)
    score = model_builder.model.evaluate(test_ds)
