import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Concatenate

def concat_image_title_model(image_model, title_model, n_labels):
    # use detailed output layer
    image_model = Model(image_model.inputs, image_model.get_layer('image_dense_1024').output) 
    image_model.trainable = False
    title_model = Model(title_model.inputs, title_model.get_layer('title_dense_1024').output)
    title_model.trainable = False
    
    concatenate = Concatenate()([image_model.output, title_model.output])
    x = Dense(1024, activation = 'relu')(concatenate)
    x = Dropout(0.4)(x)
    output = Dense(n_labels, activation='softmax')(x)
    model = Model(inputs=[image_model.input, title_model.input], outputs=output, name='concat')

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tfa.metrics.F1Score(name='f1_score', num_classes=2, average='micro')
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model
