from enum import Enum, auto
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


class ModelType(Enum):
    INCEPTION = auto()
    INCEPTIONRESNET = auto()
    EFFIECENTNET = auto()


def build_model(model_type, n_labels):    
    if model_type == ModelType.INCEPTION:
        model = __build_inception(n_labels)
    elif model_type == ModelType.INCEPTIONRESNET:
        model = __build_inceptionresnet
    elif model_type == ModelType.EFFIECENTNET:
        model = __build_effiecentnet()
    else:
        raise ValueError('Not a valid value for model')

    __compile(model, n_labels == 1)
    return model


def __build_inception(n_labels):
    base_model = InceptionV3(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(n_labels, activation='softmax')(x)

    return Model(inputs=base_model.input, outputs=predictions)


def __build_inceptionresnet():
    pass


def __build_effiecentnet():
    pass


def __compile(model, is_binary):
    loss = 'binary_crossentropy' if is_binary else 'categorical_crossentropy'
    optimizer = 'adam' if is_binary else 'rmsprop'

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
