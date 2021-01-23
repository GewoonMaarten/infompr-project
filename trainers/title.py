import tensorflow as tf
from utils.dataset_text import text_dataset
from utils.model_factory_title import build_title_model
from utils.config import training_epochs

n_labels = 2

train_seq = text_dataset('train')
test_seq = text_dataset('test')
validate_seq = text_dataset('validate')

model = build_title_model(n_labels)

callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)]
history = model.fit(train_seq, validation_data=validate_seq, epochs=training_epochs, callbacks=callbacks)
score = model.evaluate(test_seq)

print(f'Test loss: {score}')

model.save_weights("models/roberta2.hdf5")
