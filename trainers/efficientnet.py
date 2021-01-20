from utils.image_model_factory import ModelBuilder
from utils.config import training_epochs
from utils.dataset_image import DatasetImage
from utils.common import configure_for_performance
import os
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_cloud as tfc


def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


GCP_BUCKET = 'infompr-results'
MODEL_PATH = 'efficientnet'

callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)]
if tfc.remote():
    checkpoint_path = os.path.join("gs://",
                                   GCP_BUCKET, 
                                   MODEL_PATH, 
                                   "save_at_{epoch}")
    tensorboard_path = os.path.join("gs://", 
                                    GCP_BUCKET, 
                                    "logs", 
                                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(checkpoint_path))
else:
    tensorboard_path = os.path.join("logs", 
                                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
callbacks.append(tf.keras.callbacks.TensorBoard(
    log_dir=tensorboard_path, histogram_freq=1))

train_seq = configure_for_performance(DatasetImage('train'))
test_seq = configure_for_performance(DatasetImage('test'))
validate_seq = configure_for_performance(DatasetImage('validate'))

model_builder = ModelBuilder('b7')
model_builder.compile_for_transfer_learning()

print('TRANSFER LEARNING')
history = model_builder.model.fit(
    train_seq,
    validation_data=validate_seq,
    epochs=training_epochs,
    callbacks=callbacks)

if not tfc.remote():
    score = model_builder.model.evaluate(test_seq)
    print(f'Test scores: {score}')
    plot_hist(history)

print('FINE TUNING')
model_builder.compile_for_fine_tuning()
history = model_builder.model.fit(
    train_seq,
    validation_data=validate_seq,
    epochs=training_epochs,
    callbacks=callbacks)

if not tfc.remote():
    score = model_builder.model.evaluate(test_seq)
    print(f'Test scores: {score}')
    plot_hist(history)

if tfc.remote():
    SAVE_PATH = os.path.join("gs://", GCP_BUCKET, MODEL_PATH)
    model_builder.model.save(SAVE_PATH)
    model = tf.keras.models.load_model(SAVE_PATH)
else:
    model_builder.model.save("models/efficientnet.hdf5")
