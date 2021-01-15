import tensorflow as tf
from transformers import TFRobertaForSequenceClassification
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda
from tensorflow.keras.models import Model

from utils.config import (
    text_max_length)

def build_title_model(n_labels, path_to_weights=None):
    model = TFRobertaForSequenceClassification.from_pretrained("roberta-base")
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    # see https://github.com/huggingface/transformers/issues/1350
    input = Input(shape=(text_max_length,), name='input_ids', dtype='int32')
    output = model(input)[0]
    # Keep [CLS] token encoding
    # output = Lambda(lambda seq: seq[:, 0], name = 'title_dense_1024')(output)

    # output = tf.squeeze(output[:, 0:1], axis=1)
    output = Dense(1024, activation='relu', name='title_dense_1024')(output)
    output = Dropout(0.5, name = 'title_dropout')(output) 
    output = Dense(n_labels, activation="softmax", name = 'title_softmax')(output)

    model = Model(inputs=[input], outputs=[output], name='title_model')

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    return model