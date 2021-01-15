from enum import Enum, auto
from pathlib import Path
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as inception_resnet_preprocess_input
import cv2
import math
import numpy as np
import pandas as pd

from utils.config import (
    dataset_test_path,
    dataset_train_path,
    dataset_validate_path,
    dataset_images_path,
    img_size,
    training_batch_size)
from utils.image_model_factory import ModelType

class ModeType(Enum):
    TRAIN = auto()
    TEST = auto()
    VALIDATE = auto()


class ImageSequence(Sequence):
    """ Sequence for loading data and training models """

    def __init__(self, model_type, mode=ModeType.TRAIN, n_labels=2):
        """ Creates an instance of ImageSequence

        Parameters:
        -----------
            batch_size (int): size of the batch to return with `__getitem__()`
            image_size ((int, int)): size of the image
            mode (ModeType): for which purpose is the sequence. This value determines
            from which dataset to load the samples.
            n_labels (int): number of labels to train on. Can be 2, 3 or 6.
        """
        df_path = None
        if mode == ModeType.TRAIN:
            df_path = dataset_train_path
        elif mode == ModeType.TEST:
            df_path = dataset_test_path
        elif mode == ModeType.VALIDATE:
            df_path = dataset_validate_path
        else:
            raise ValueError(
                'mode can only be \'train\', \'test\' or \'validate\'')

        if not n_labels in (2, 3, 6):
            raise ValueError('n_labels can only be 2, 3, or 6')

        self.df = pd.read_csv(df_path, sep='\t', header=0)
        self.batch_size = training_batch_size
        self.image_size = img_size
        self.model_type = model_type
        self.n_labels = n_labels
        self.labels = f'{n_labels}_way_label'

        self.on_epoch_end()  # initial shuffle

    def __len__(self):
        """ Denotes the number of batches per epoch """
        return math.floor(len(self.df) / self.batch_size)

    def __getitem__(self, index):
        """ Generate one batch of data """
        df_slice = self.df.iloc[index * self.batch_size:
                                (index+1) * self.batch_size]

        return np.array([
            self.__load_img(id) for id in df_slice.id]), \
            to_categorical(df_slice[self.labels], self.n_labels)

    def on_epoch_end(self):
        """ Shuffle df after each epoch """
        self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __load_img(self, id):
        """ Load image as 1d array """
        path = Path(dataset_images_path, f'{id}.jpg')
        # load in color mode as (x,y,3) array
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        # resize with resampling using pixel area relation
        img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)

        if self.model_type == ModelType.INCEPTION:
            img = inception_preprocess_input(img)
        elif self.model_type == ModelType.INCEPTIONRESNET:
            img = inception_resnet_preprocess_input(img)
        elif self.model_type == ModelType.EFFIECENTNET:
            img = efficientnet_preprocess_input(img)
        
        return img
