import tensorflow as tf
from utils.dataset_text import TextDataset
from utils.model_factory_title import build_title_model
from utils.config import training_epochs, text_use_bert

n_labels = 2

train_seq = TextDataset('train', text_use_bert).text_dataset()
test_seq = TextDataset('test', text_use_bert).text_dataset()
validate_seq = TextDataset('validate', text_use_bert).text_dataset()

model = build_title_model(n_labels, text_use_bert)

callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)]
history = model.fit(train_seq, validation_data=validate_seq, epochs=training_epochs, callbacks=callbacks)
score = model.evaluate(test_seq)

print(f'Test loss: {score}')

model.save_weights("models/Text_roBERTa_10K_V3.hdf5")
