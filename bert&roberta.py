from transformers import RobertaTokenizer, BertTokenizer
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import TFRobertaForSequenceClassification, TFBertForSequenceClassification
import time

from utils.config import (
    dataset_test_path,
    dataset_train_path,
    dataset_validate_path)

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))  

def read_data(data, n_labels):
    df = pd.read_csv(data, sep='\t', header =0)
    X_train = df['clean_title']
    y_train = df[n_labels]
    return df, X_train, y_train

df_train, X_train, y_train = read_data(dataset_train_path, n_labels= "2_way_label")
df_val, X_val, y_val = read_data(dataset_validate_path, n_labels= "2_way_label")
df_test, X_test, y_test = read_data(dataset_test_path, n_labels= "2_way_label")


roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

model_roberta = TFRobertaForSequenceClassification.from_pretrained("roberta-base")
model_bert = bert_model = TFBertForSequenceClassification.from_pretrained("bert-base-cased")


#Defining the hyper-parameters
number_of_epochs = 3
max_length = 128
batch_size = 64
learning_rate=3e-5
num_labels = 2

## code source: https://colab.research.google.com/drive/1l39vWjZ5jRUimSQDoUcuWGIoNjLjA2zu#scrollTo=scT82c9arCRv
## code source: https://towardsdatascience.com/discover-the-sentiment-of-reddit-subgroup-using-roberta-model-10ab9a8271b8
def convert_example_to_feature(review, roberta):
    # combine step for tokenization, WordPiece vector mapping and will
    # add also special tokens and truncate reviews longer than our max length
    if roberta: 
        return roberta_tokenizer.encode_plus(review,
                                     add_special_tokens=True,  # add [CLS], [SEP]
                                     max_length=max_length,  # max length of the text that can go to RoBERTa
                                     pad_to_max_length=True,  # add [PAD] tokens at the end of sentence
                                     return_attention_mask=True,  # add attention mask to not focus on pad tokens
                                     )

    # combine step for tokenization, WordPiece vector mapping and will
    # add also special tokens and truncate reviews longer than our max length
    return bert_tokenizer.encode_plus(review,
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

def encode_examples(ds, roberta, limit=-1):
    # Prepare Input list
    input_ids_list = []
    attention_mask_list = []
    label_list = []

    if (limit > 0):
        ds = ds.take(limit)

    for review, label in tfds.as_numpy(ds):
        bert_input = convert_example_to_feature(review.decode(), roberta)
        input_ids_list.append(bert_input['input_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append([label])

    return tf.data.Dataset.from_tensor_slices((input_ids_list,
                                               attention_mask_list,
                                               label_list)).map(map_example_to_dict)


# create dataset suitable for roBERTa model
training_sentences_modified = tf.data.Dataset.from_tensor_slices((X_train[:300], y_train[:300]))
validating_sentences_modified = tf.data.Dataset.from_tensor_slices((X_val, y_val))
testing_sentences_modified = tf.data.Dataset.from_tensor_slices((X_test, y_test))

  

def run_model(roberta):
    ds_train_encoded = encode_examples(training_sentences_modified, roberta = roberta).shuffle(100).batch(batch_size)
    ds_val_encoded = encode_examples(validating_sentences_modified, roberta = roberta).batch(batch_size)
    ds_test_encoded = encode_examples(testing_sentences_modified, roberta = roberta).batch(batch_size)


    # hyperparameters BERT
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08, clipnorm=1.0)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    if roberta:
        model = model_roberta
    else:
        model = model_bert
    
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    
    #train the model
    model.fit(ds_train_encoded, epochs=number_of_epochs,
              validation_data=ds_val_encoded)
    
    # evaluate the performance on the validation set
    print(model.evaluate(ds_val_encoded))
    print(model.evaluate(ds_test_encoded))
    return model

print("roBERTa")
roberta = run_model(True)
print("BERT")
bert = run_model(False)

end = time.time()

