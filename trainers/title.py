import tensorflow as tf
from utils.dataset_text import DatasetText
from utils.model_factory_title import build_title_model
from utils.config import (training_epochs)
from utils.common import configure_for_performance

n_labels = 2

train_seq = configure_for_performance(DatasetText('train'))
test_seq = configure_for_performance(DatasetText('test'))
validate_seq = configure_for_performance(DatasetText('validate'))

callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)]

model = build_title_model(n_labels)

history = model.fit(train_seq, validation_data=validate_seq, epochs=training_epochs, callbacks=callbacks)
score = model.evaluate(test_seq)

print(f'Test loss: {score}')

model.save_weights("models/roberta2.hdf5")
