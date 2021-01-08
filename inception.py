from utils.data_sequence import FakedditSequence
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

batch_size = 64
image_size = (299, 299)
train_seq = FakedditSequence(batch_size, image_size, mode='train')
test_seq = FakedditSequence(batch_size, image_size, mode='test')
validate_seq = FakedditSequence(batch_size, image_size, mode='validate')


base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(train_seq, validation_data=validate_seq)
