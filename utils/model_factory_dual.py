import tensorflow as tf
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
    model = Model(inputs=[image_model.input, title_model.input], outputs=output, name = 'concat')

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    return model
