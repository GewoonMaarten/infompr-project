from pathlib import Path
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import (
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB2,
    EfficientNetB3,
    EfficientNetB4,
    EfficientNetB5,
    EfficientNetB6,
    EfficientNetB7)
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from utils.config import img_height, img_width

SIZES = {
    'b0': {
        'model': EfficientNetB0,
        'weights': 'efficientnetb0_notop.h5'
    },
    'b1': {
        'model': EfficientNetB1,
        'weights': 'efficientnetb1_notop.h5'
    },
    'b2': {
        'model': EfficientNetB2,
        'weights': 'efficientnetb2_notop.h5'
    },
    'b3': {
        'model': EfficientNetB3,
        'weights': 'efficientnetb3_notop.h5'
    },
    'b4': {
        'model': EfficientNetB4,
        'weights': 'efficientnetb4_notop.h5'
    },
    'b5': {
        'model': EfficientNetB5,
        'weights': 'efficientnetb5_notop.h5'
    },
    'b6': {
        'model': EfficientNetB6,
        'weights': 'efficientnetb6_notop.h5'
    },
    'b7': {
        'model': EfficientNetB7,
        'weights': 'efficientnetb7_notop.h5'
    }
}


class ModelBuilder():
    def __init__(self, size) -> None:
        self.model = None
        self.base_model = None
        self.__build_efficientnet(SIZES[size]['model'], SIZES[size]['weights'])
        self.metrics = [
            tf.keras.metrics.BinaryAccuracy('accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tfa.metrics.F1Score(num_classes=2, average='macro', name='f1_score')
        ]

    def compile_for_transfer_learning(self):
        self.base_model.trainable = False
        optimizer = Adam(learning_rate=1e-2)
        self.model.compile(optimizer=optimizer,
                           loss='binary_crossentropy',
                           metrics=self.metrics)

    def compile_for_fine_tuning(self):
        self.base_model.trainable = True
        # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
        for layer in self.model.layers[-20:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True

        optimizer = Adam(learning_rate=1e-4)
        self.model.compile(optimizer=optimizer,
                           loss='binary_crossentropy',
                           metrics=self.metrics)

    def __build_efficientnet(self, model, weight_path):
        weights_path = Path(Path(__file__).parent.parent.absolute(),
                            'bin',
                            'noisy-student',
                            weight_path)

        if not weights_path.exists():
            raise Exception('Could not find weights file?!')

        input_tensor = layers.Input(shape=(img_width, img_height, 3))
        self.base_model = model(input_tensor=input_tensor,
                                weights='imagenet',
                                include_top=False)

        x = self.base_model.output
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        x = layers.BatchNormalization(name='image_dense_1024')(x)
        x = layers.Dropout(0.5, name="top_dropout")(x)

        predictions = layers.Dense(2, activation="softmax")(x)

        self.model = Model(inputs=self.base_model.input,
                           outputs=predictions,
                           name="EfficientNet")
