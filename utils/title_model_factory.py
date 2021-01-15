import tensorflow as tf
from transformers import TFRobertaForSequenceClassification
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

def build_title_model(n_labels, path_to_weights=None):
    model = TFRobertaForSequenceClassification.from_pretrained("roberta-base")
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    # see https://github.com/huggingface/transformers/issues/1350
    input = Input(shape=(None,), name='word_inputs', dtype='int32')
    output = model(input)[0]
    # Keep [CLS] token encoding
    # doc_encoding = tf.squeeze(roberta_encodings[:, 0], axis=1)

    output = Dense(1024, activation='relu', name='title_dense_1024')(output)
    output = Dropout(0.5)(output) 
    output = Dense(n_labels, activation="softmax")(output)

    model = Model(inputs=[input], outputs=[output])

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    return model