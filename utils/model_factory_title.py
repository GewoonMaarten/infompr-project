import tensorflow as tf
from transformers import (
    TFRobertaModel, 
    TFBertModel,
    RobertaConfig,
    BertConfig)

from utils.config import text_max_length,text_use_bert

def build_title_model(n_labels):

    if text_use_bert:
        config = BertConfig(output_hidden_states=True) # I dont know why config doesnt work
        transformer_model = TFBertModel.from_pretrained("bert-base-cased")
    else:
        config = RobertaConfig(output_hidden_states=True)
        transformer_model = TFRobertaModel.from_pretrained("roberta-base")
    
    input_ids_in = tf.keras.layers.Input(shape=(text_max_length,), name='input_token', dtype=tf.int32)
    input_masks_in = tf.keras.layers.Input(shape=(text_max_length,), name='masked_token', dtype=tf.int32)

    embedding_layer = transformer_model(input_ids_in, attention_mask=input_masks_in)[0]
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, return_sequences=True, dropout=0.1))(embedding_layer)
    x = tf.keras.layers.GlobalMaxPool1D()(x)
    x = tf.keras.layers.Dense(50, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2, name='title_dense_1024')(x)
    predictions = tf.keras.layers.Dense(2, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=[input_ids_in, input_masks_in], outputs=predictions)

    for layer in model.layers[:3]:
        layer.trainable = False

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.CategoricalAccuracy('accuracy')

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    return model
