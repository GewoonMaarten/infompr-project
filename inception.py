from utils.data_sequence import FakedditSequence
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

from utils.config import (
    training_batch_size,
    training_epochs,
    img_size
)

train_seq = FakedditSequence(training_batch_size, img_size, mode='train')
test_seq = FakedditSequence(training_batch_size, img_size, mode='test')
validate_seq = FakedditSequence(training_batch_size, img_size, mode='validate')


base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(train_seq, validation_data=validate_seq, epochs=training_epochs)
score = model.evaluate(test_seq)

print(f'Test loss: {score}')
