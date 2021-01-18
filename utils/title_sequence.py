from enum import Enum, auto
from pathlib import Path
from tensorflow.keras.utils import Sequence, to_categorical
import cv2
import math
import numpy as np
import pandas as pd
from .data_sequence import ModeType
from transformers import RobertaTokenizer

from utils.config import (
    dataset_test_path,
    dataset_train_path,
    dataset_validate_path,
    dataset_images_path,
    text_max_length)

roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
def convert_example_to_feature(review):
    # combine step for tokenization, WordPiece vector mapping and will
    # add also special tokens and truncate reviews longer than our max length
    return roberta_tokenizer.encode_plus(review,
                                 add_special_tokens=True,  # add [CLS], [SEP]
                                 max_length=text_max_length,  # max length of the text that can go to RoBERTa
                                 pad_to_max_length=True,  # add [PAD] tokens at the end of sentence
                                 return_attention_mask=True,  # add attention mask to not focus on pad tokens
                                 )

class FakedditTitleSequence(Sequence):
    """ Sequence for loading data and training models """

    def __init__(self, batch_size, image_size, mode=ModeType.TRAIN, n_labels=2):
        """ Creates an instance of FakedditSequence

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
        self.batch_size = batch_size
        self.image_size = image_size
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

        titles = np.array([convert_example_to_feature(title)['input_ids'] for title in df_slice['clean_title']])
        # labels = to_categorical(df_slice[self.labels], self.n_labels)
        labels = np.array(df_slice[self.labels]).astype(np.float32)
        return titles, labels

    def on_epoch_end(self):
        """ Shuffle df after each epoch """
        self.df = self.df.sample(frac=1).reset_index(drop=True)
