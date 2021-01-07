import pandas as pd
import math
import cv2
from pathlib import Path
import numpy as np
from tensorflow.keras.utils import Sequence

from utils.config import (
    dataset_test_path,
    dataset_train_path,
    dataset_validate_path,
    dataset_images_path)


class FakedditSequence(Sequence):
    """ Sequence for loading data and training models """

    def __init__(self, batch_size, image_size, mode='train', n_labels=2):
        """ Creates an instance of FakedditSequence

        Parameters:
        -----------
            batch_size (int): size of the batch to return with `__getitem__()`
            image_size ((int, int)): size of the image
            mode (str): for which purpose is the sequence. This value determines
            from which dataset to load the samples. Can have the following
            values:
                * 'train'
                * 'test'
                * 'validate'
            n_labels (int): number of labels to train on. Can be 2, 3 or 6.
        """
        df_path = None
        if mode == 'train':
            df_path = dataset_train_path
        elif mode == 'test':
            df_path = dataset_test_path
        elif mode == 'validate':
            df_path = dataset_validate_path
        else:
            raise ValueError(
                'mode can only be \'train\', \'test\' or \'validate\'')

        if not n_labels in (2, 3, 6):
            raise ValueError('n_labels can only be 2, 3, or 6')

        self.df = pd.read_csv(df_path, sep='\t', header=0)
        self.batch_size = batch_size
        self.image_size = image_size
        self.labels = f'{n_labels}_way_label'

        self.on_epoch_end() # initial shuffle

    def __len__(self):
        """ Denotes the number of batches per epoch """
        return math.floor(len(self.df) / self.batch_size)

    def __getitem__(self, index):
        """ Generate one batch of data """
        df_slice = self.df.iloc[index * self.batch_size:
                                (index+1) * self.batch_size]

        return np.array([
            self._load_img(id) for id in df_slice.id]), \
            self.df[self.labels].to_numpy()

    def on_epoch_end(self):
        """ Shuffle df after each epoch """
        self.df = self.df.sample(frac=1).reset_index(drop=True)

    def _load_img(self, id):
        """ Load image as 1d array """
        path = Path(dataset_images_path, f'{id}.jpg')
        # load in color mode as (x,y,3) array
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        # resize with resampling using pixel area relation
        img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
        # flatten img array 3d -> 1d
        img = img.flatten()

        return img
