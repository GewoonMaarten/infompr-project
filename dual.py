from utils.dataset_dual import DualDataset, configure_for_performance
from utils.title_model_factory import build_title_model
from utils.image_model_factory import ModelBuilder
from utils.dual_model_factory import concat_image_title_model
from utils.config import (
    training_batch_size,
    training_epochs,
    img_size
)

n_labels = 2

train_seq = configure_for_performance(DualDataset('train'))
test_seq = configure_for_performance(DualDataset('test'))
validate_seq = configure_for_performance(DualDataset('validate'))

image_model = ModelBuilder('b0')
image_model.compile_for_transfer_learning()

image_model = image_model.model
image_model.load_weights("models/efficientnet.hdf5")

image_model.trainable = False
# for layer in image_model.layers:
#     layer.trainable = False
title_model = build_title_model(n_labels, "models/roberta.hdf5")

model = concat_image_title_model(image_model, title_model, n_labels)

history = model.fit(train_seq, validation_data=validate_seq, epochs=training_epochs)
score = model.evaluate(test_seq)

print(f'Test loss: {score}')

model.save_weights("models/dual.hdf5")
