from utils.data_sequence import ModeType
from utils.title_sequence import FakedditTitleSequence
from utils.title_model_factory import build_title_model
from utils.config import (
    training_batch_size,
    training_epochs,
    img_size
)

n_labels = 2

train_seq = FakedditTitleSequence(
    training_batch_size, 
    img_size, 
    mode=ModeType.TRAIN, 
    n_labels=n_labels)

test_seq = FakedditTitleSequence(
    training_batch_size,
    img_size, 
    mode=ModeType.TEST,
     n_labels=n_labels)
    
validate_seq = FakedditTitleSequence(
    training_batch_size, 
    img_size, 
    mode=ModeType.VALIDATE, 
    n_labels=n_labels)

model = build_title_model(n_labels)

history = model.fit(train_seq, validation_data=validate_seq, epochs=training_epochs)
score = model.evaluate(test_seq)

print(f'Test loss: {score}')

model.save_weights("models/roberta2.hdf5")
