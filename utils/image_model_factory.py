from enum import Enum, auto
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.layers import Input, BatchNormalization, Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from pathlib import Path

from utils.config import img_height, img_width

class ModelType(Enum):
    INCEPTION = auto()
    INCEPTIONRESNET = auto()
    EFFIECENTNET = auto()


<<<<<<< HEAD
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
=======
class ModelBuilder():
    def __init__(self, model_type, n_labels) -> None:
        self.model_type = model_type
        self.n_lables = n_labels
        self.model = None
        self.base_model = None
        
        if model_type == ModelType.INCEPTION:
            self.__build_inception(n_labels)
        elif model_type == ModelType.INCEPTIONRESNET:
            self.__build_inceptionresnet(n_labels)
        elif model_type == ModelType.EFFIECENTNET:
            self.__build_effiecentnet(n_labels)
        else:
            raise ValueError('Not a valid value for model')


    def __build_inception(self, n_labels):
        input_tensor = Input(shape=(img_width, img_height, 3))
        self.base_model = InceptionV3(input_tensor=input_tensor, 
                                      weights='imagenet', 
                                      include_top=False)

        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)

        predictions = Dense(n_labels, activation='softmax')(x)

        self.model = Model(inputs=self.base_model.input,
                           outputs=predictions, 
                           name="Inception")

    def __build_inceptionresnet(self, n_labels):
        input_tensor = Input(shape=(img_width, img_height, 3))
        self.base_model = InceptionResNetV2(input_tensor=input_tensor, 
                                            weights='imagenet', 
                                            include_top=False)

        x = self.base_model.output
        x = Flatten()(x)
        x = Dropout(0.5)(x)

        predictions = Dense(n_labels, activation='softmax')(x)

        self.model = Model(inputs=self.base_model.input,
                           outputs=predictions, 
                           name="Inception-ResNet")

    def __build_effiecentnet(self, n_labels):
        weights_path = Path(Path().absolute(),
                            'bin',
                            'noisy-student',
                            'efficientnetb7_notop.h5')

        if not weights_path.exists():
            raise Exception('Could not find weights file?!')

        input_tensor = Input(shape=(img_width, img_height, 3))
        self.base_model = EfficientNetB7(input_tensor=input_tensor,
                                         weights=str(weights_path), 
                                         include_top=False)

        x = self.base_model.output
        x = GlobalAveragePooling2D(name="avg_pool")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2, name="top_dropout")(x)

        predictions = Dense(n_labels, activation="softmax")(x)

        self.model = Model(inputs=self.base_model.input,
                           outputs=predictions, 
                           name="EfficientNet")

    def compile_for_transfer_learning(self):
        self.base_model.trainable = False
        loss = 'binary_crossentropy' if self.n_lables == 2 else 'categorical_crossentropy'
        self.model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

    def compile_for_fine_tuning(self):
<<<<<<< HEAD
        pass
>>>>>>> 1e95774 (Refactor image model factory)
=======
        for layer in self.model.layers[:249]:
            layer.trainable = False
        for layer in self.model.layers[249:]:
            layer.trainable = True
        # self.base_model.trainable = False
        loss = 'binary_crossentropy' if self.n_lables == 2 else 'categorical_crossentropy'
        self.model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), 
                           loss=loss, 
                           metrics=['accuracy'])
>>>>>>> a0c5edc (Add fine tuning to image model factory)
