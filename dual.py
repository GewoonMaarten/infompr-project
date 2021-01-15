from utils.data_sequence import ModeType
from utils.dual_sequence import FakedditDualSequence
from utils.title_model_factory import build_title_model
from utils.image_model_factory import build_model as build_image_model, ModelType
from utils.dual_model_factory import concat_image_title_model
from utils.config import (
    training_batch_size,
    training_epochs,
    img_size
)

n_labels = 2

train_seq = FakedditDualSequence(
    training_batch_size, 
    img_size, 
    mode=ModeType.TRAIN, 
    n_labels=n_labels)

test_seq = FakedditDualSequence(
    training_batch_size,
    img_size, 
    mode=ModeType.TEST,
     n_labels=n_labels)
    
validate_seq = FakedditDualSequence(
    training_batch_size, 
    img_size, 
    mode=ModeType.VALIDATE, 
    n_labels=n_labels)

image_model = build_image_model(ModelType.INCEPTION, n_labels=n_labels, path_to_weights="models/inception3.hdf5")
title_model = build_title_model(n_labels, "models/roberta.hdf5")

model = concat_image_title_model(image_model, title_model, n_labels)

history = model.fit(train_seq, validation_data=validate_seq, epochs=training_epochs)
score = model.evaluate(test_seq)

print(f'Test loss: {score}')

model.save("models/dual.hdf5")
