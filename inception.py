from utils.data_sequence import FakedditSequence
from utils.image_model_factory import build_model, ModelType
from utils.config import (
    training_batch_size,
    training_epochs,
    img_size
)
n_labels = 2
train_seq = FakedditSequence(training_batch_size, img_size, mode='train', n_labels=n_labels)
test_seq = FakedditSequence(training_batch_size, img_size, mode='test', n_labels=n_labels)
validate_seq = FakedditSequence(training_batch_size, img_size, mode='validate', n_labels=n_labels)

model = build_model(ModelType.INCEPTION, n_labels=n_labels)

model.fit(train_seq, validation_data=validate_seq, epochs=training_epochs)
score = model.evaluate(test_seq)

print(f'Test loss: {score}')
