from enum import Enum, auto
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from pathlib import Path


class ModelType(Enum):
    INCEPTION = auto()
    INCEPTIONRESNET = auto()
    EFFIECENTNET = auto()


def build_model(model_type, n_labels, path_to_weights=None):
    if model_type == ModelType.INCEPTION:
        model = __build_inception(n_labels, path_to_weights)
    elif model_type == ModelType.INCEPTIONRESNET:
        model = __build_inceptionresnet(n_labels, path_to_weights)
    elif model_type == ModelType.EFFIECENTNET:
        model = __build_effiecentnet(n_labels, path_to_weights)
    else:
        raise ValueError('Not a valid value for model')

    __compile(model, n_labels == 1)
    return model


def __build_inception(n_labels, path_to_weights=None):
    base_model = InceptionV3(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu', name='image_dense_1024')(x)

    predictions = Dense(n_labels, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions, name="Inception")

    if path_to_weights:
        model.load_weights(path_to_weights)

    return model


def __build_inceptionresnet(n_labels, path_to_weights=None):
    base_model = InceptionResNetV2(weights='imagenet', include_top=False)

    x = base_model.output
    x = Flatten()(x)
    x = Dropout(0.5)(x)

    predictions = Dense(n_labels, activation='softmax')(x)

    for layer in base_model.layers:
        layer.trainable = False

    return Model(inputs=base_model.input, outputs=predictions, name="Inception-ResNet")


def __build_effiecentnet(n_labels, path_to_weights=None):
    weights_path = Path(Path().absolute(),
                        'bin',
                        'noisy-student',
                        'efficientnetb7_notop.h5')

    if not weights_path.exists:
        raise Exception('Could not find weights file?!')

    base_model = EfficientNetB7(weights=str(weights_path), include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D(name="avg_pool")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2, name="top_dropout")(x)

    predictions = Dense(n_labels, activation="softmax")(x)

    for layer in base_model.layers:
        layer.trainable = False

    return Model(inputs=base_model.input, outputs=predictions, name="EfficientNet")


def __compile(model, is_binary):

    loss = 'binary_crossentropy' if is_binary else 'categorical_crossentropy'
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
