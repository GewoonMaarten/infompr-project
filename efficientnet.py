from utils.image_model_factory import ModelBuilder
from utils.config import training_epochs
from utils.image_dataset import ImageDataset, configure_for_performance
import os
import matplotlib.pyplot as plt

def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

train_seq = configure_for_performance(ImageDataset('train'))
test_seq = configure_for_performance(ImageDataset('test'))
validate_seq = configure_for_performance(ImageDataset('validate'))

model_builder = ModelBuilder('b0')
model_builder.compile_for_transfer_learning()

history = model_builder.model.fit(train_seq, validation_data=validate_seq, epochs=training_epochs)
score = model_builder.model.evaluate(test_seq)
print(f'Test scores: {score}')
plot_hist(history)

model_builder.compile_for_fine_tuning()
history = model_builder.model.fit(train_seq, validation_data=validate_seq, epochs=training_epochs)
score = model_builder.model.evaluate(test_seq)
print(f'Test scores: {score}')
plot_hist(history)

# model_builder.model.save("models/inception3.hdf5")
