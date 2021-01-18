from transformers import RobertaTokenizer, BertTokenizer
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import TFRobertaForSequenceClassification, TFBertForSequenceClassification
from utils.config import (
    dataset_test_path,
    dataset_train_path,
    dataset_validate_path)


def read_data(data, n_labels = "2_way_label"):
    df = pd.read_csv(data, sep='\t', header =0)
    X_train = df['clean_title']
    y_train = df[n_labels]
    return df, X_train, y_train


df_train, X_train_s, y_train_s = read_data(dataset_train_path)
df_val, X_val_s, y_val_s = read_data(dataset_validate_path)
df_test, X_test_s, y_test_s = read_data(dataset_test_path)


roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

model_roberta = TFRobertaForSequenceClassification.from_pretrained("roberta-base")
model_bert = bert_model = TFBertForSequenceClassification.from_pretrained("bert-base-cased")


#Defining the hyper-parameters
number_of_epochs = 8
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
training_sentences_modified = tf.data.Dataset.from_tensor_slices((X_train_s[:300], y_train_s[:300]))
validating_sentences_modified = tf.data.Dataset.from_tensor_slices((X_val_s, y_val_s))
testing_sentences_modified = tf.data.Dataset.from_tensor_slices((X_test_s, y_test_s))

  

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



def plot_token_length(X_train):
    token_length = []
    normal_word_len = []
    ids = []
    for i in range(len(X_train)):
        length = len(roberta_tokenizer.tokenize(X_train[i]))
        if length > 128:
            token_length.append(length)
            ids.append(df_train['id'][i])
            normal_word_len.append(X_train[i])
    return token_length, normal_word_len, ids

"""
## 
input_layer = Input(shape = (512,), dtype='int64')
bert = TFBertModel.from_pretrained('bert-base-cased')(input_layer)
bert = bert[0]              # i think there is a bug here
flat = Flatten()(bert)
classifier = Dense(units=5)(flat)
model = Model(inputs=input_layer, outputs=classifier)
model.summary()
"""
""""""
