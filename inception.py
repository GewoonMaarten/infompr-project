from utils.data_sequence import FakedditSequence, ModeType
from utils.image_model_factory import ModelBuilder, ModelType
from utils.config import (
    training_batch_size,
    training_epochs,
    img_size
)

n_labels = 2

train_seq = FakedditSequence(
    training_batch_size,
    img_size,
    mode=ModeType.TRAIN,
    n_labels=n_labels)

test_seq = FakedditSequence(
    training_batch_size,
    img_size,
    mode=ModeType.TEST,
    n_labels=n_labels)

validate_seq = FakedditSequence(
    training_batch_size,
    img_size,
    mode=ModeType.VALIDATE,
    n_labels=n_labels)

model_builder = ModelBuilder(ModelType.INCEPTION, n_labels=n_labels)
model_builder.compile_for_transfer_learning()

history = model_builder.model.fit(train_seq, validation_data=validate_seq, epochs=training_epochs)
score = model_builder.model.evaluate(test_seq)

print(f'Test scores: {score}')

model_builder.compile_for_fine_tuning()

history = model_builder.model.fit(train_seq, validation_data=validate_seq, epochs=training_epochs)
score = model_builder.model.evaluate(test_seq)

print(f'Test scores: {score}')

model.save("models/inception3.hdf5")
