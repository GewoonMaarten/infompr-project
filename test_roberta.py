from transformers import RobertaTokenizer, TFRobertaModel
import pandas as pd
import  numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import TFRobertaForSequenceClassification

from utils.config import (
    dataset_test_path,
    dataset_train_path,
    dataset_validate_path)

def read_data(data, n_labels):
    df = pd.read_csv(data, sep='\t', header =0)
    X_train = df['clean_title']
    y_train = df[n_labels]
    return df, X_train, y_train

    
df_train, X_train, y_train = read_data(dataset_train_path, n_labels= "2_way_label")
df_val, X_val, y_val = read_data(dataset_validate_path, n_labels= "2_way_label")
df_test, X_test, y_test = read_data(dataset_test_path, n_labels= "2_way_label")



roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# can be up  512 for BERT
max_length = 100
batch_size = 64

def convert_example_to_feature(review):
    # combine step for tokenization, WordPiece vector mapping and will
    # add also special tokens and truncate reviews longer than our max length
    return roberta_tokenizer.encode_plus(review,
                                 add_special_tokens=True,  # add [CLS], [SEP]
                                 max_length=max_length,  # max length of the text that can go to RoBERTa
                                 pad_to_max_length=True,  # add [PAD] tokens at the end of sentence
                                 return_attention_mask=True,  # add attention mask to not focus on pad tokens
                                 )

# map to the expected input to TFRobertaForSequenceClassification, see here
def map_example_to_dict(input_ids, attention_masks, label):
    return {
      "input_ids": input_ids,
      "attention_mask": attention_masks,
           }, label

def encode_examples(ds, limit=-1):
    # Prepare Input list
    input_ids_list = []
    attention_mask_list = []
    label_list = []

    if (limit > 0):
        ds = ds.take(limit)

    for review, label in tfds.as_numpy(ds):
        bert_input = convert_example_to_feature(review.decode())
        input_ids_list.append(bert_input['input_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append([label])

    return tf.data.Dataset.from_tensor_slices((input_ids_list,
                                               attention_mask_list,
                                               label_list)).map(map_example_to_dict)

training_sentences_modified = tf.data.Dataset.from_tensor_slices((X_train, y_train))
testing_sentences_modified = tf.data.Dataset.from_tensor_slices((X_val, y_val))
                      

ds_train_encoded = encode_examples(training_sentences_modified).shuffle(100).batch(batch_size)
ds_test_encoded = encode_examples(testing_sentences_modified).batch(batch_size)

learning_rate = 7e-5
number_of_epochs = 3


model = TFRobertaForSequenceClassification.from_pretrained("roberta-base")
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')


model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

model.fit(ds_train_encoded, epochs=number_of_epochs,
          validation_data=ds_test_encoded)

model.evaluate(ds_test_encoded)
    
