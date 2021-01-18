from utils.image_model_factory import ModelBuilder
from utils.config import training_epochs
from utils.dataset_image import ImageDataset, configure_for_performance
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
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# tfc.run(
#     requirements_txt="requirements.txt",
#     distribution_strategy="auto",
#     chief_config=tfc.MachineConfig(
#         cpu_cores=8,
#         memory=30,
#         accelerator_type=tfc.AcceleratorType.NVIDIA_TESLA_T4,
#         accelerator_count=1,
#     ),
#     worker_count=0,
#     stream_logs=True
#     # docker_image_bucket_name=GCP_BUCKET,
# )

checkpoint_path = os.path.join(
    "gs://", GCP_BUCKET, MODEL_PATH, "save_at_{epoch}")
tensorboard_path = os.path.join(
    "gs://", GCP_BUCKET, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)

if tfc.remote():
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path),
        tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_path, histogram_freq=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
    ]
else:
    callbacks = None

train_seq = configure_for_performance(ImageDataset('train'))
test_seq = configure_for_performance(ImageDataset('test'))
validate_seq = configure_for_performance(ImageDataset('validate'))

model_builder = ModelBuilder('b0')
model_builder.compile_for_transfer_learning()

history = model_builder.model.fit(train_seq, validation_data=validate_seq, epochs=training_epochs, callbacks=callbacks)
score = model_builder.model.evaluate(test_seq)
print(f'Test scores: {score}')
# plot_hist(history)

model_builder.compile_for_fine_tuning()
history = model_builder.model.fit(train_seq, validation_data=validate_seq, epochs=training_epochs, callbacks=callbacks)
score = model_builder.model.evaluate(test_seq)
print(f'Test scores: {score}')
# plot_hist(history)

if tfc.remote():
    SAVE_PATH = os.path.join("gs://", GCP_BUCKET, MODEL_PATH)
    model_builder.model.save(SAVE_PATH)
    model = tf.keras.models.load_model(SAVE_PATH)
#     # model.evaluate(test_ds)

model_builder.model.save("models/efficientnet.hdf5")
