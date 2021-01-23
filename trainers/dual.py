from utils.dataset_dual import DatasetDual
from utils.model_factory_title import build_title_model
from utils.model_factory_image import ModelBuilder
from utils.model_factory_dual import concat_image_title_model
from utils.config import training_epochs
from utils.common import configure_for_performance

import tensorflow as tf

n_labels = 2
image_model_name = 'EfficientNET_B3_10K_imagenet_V1'
text_model_name = 'roBERTa_10K_v1'
model_name = 'dual_10K_roBERTa_EfficientNET_B3_imagenet_V1'

train_seq = configure_for_performance(DatasetDual('train'))
test_seq = configure_for_performance(DatasetDual('test'))
validate_seq = configure_for_performance(DatasetDual('validate'))

image_model = ModelBuilder('b3')
image_model.compile_for_transfer_learning()

image_model = image_model.model
image_model.load_weights("models/efficientnet.hdf5")
image_model.trainable = False

title_model = build_title_model(n_labels)
title_model.load_weights("models/roberta.hdf5")
title_model.trainable = False

model = concat_image_title_model(image_model, title_model, n_labels)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='checkpoints/' + model_name + '_{epoch:02d}.hdf5', 
        save_weights_only=True)]

history = model.fit(
    train_seq, 
    validation_data=validate_seq, 
    epochs=training_epochs, 
    callbacks=callbacks)
score = model.evaluate(test_seq)

print(f'Test loss: {score}')

model.save_weights(f"models/{model_name}.hdf5")
