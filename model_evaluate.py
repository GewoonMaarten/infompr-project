import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from utils.model_factory_image import ModelBuilder
from utils.model_factory_title import build_title_model
from utils.model_factory_dual import concat_image_title_model

from utils.dataset_image import image_dataset
from utils.dataset_text import text_dataset, text_dataset_basic
from utils.dataset_dual import dual_dataset

from utils.config import dataset_test_path, training_batch_size

image_model_name = 'EfficientNET_B3_10K_noisy_student_V2'
dual_model_name = 'Dual_10K_roBERTa_EfficientNET_B3_noisy_student_V2'

# image_model = ModelBuilder('b3')
# image_model.compile_for_transfer_learning()
# image_model.model.load_weights(f"models/{image_model_name}.hdf5")
# image_model = image_model.model

# dual_model = concat_image_title_model(image_model, title_model, 2)
# dual_model.load_weights(f'models/{dual_model_name}.hdf5')

# image_dataset = image_dataset('test')

# dual_dataset = dual_dataset('test')

# ------------------------------------------------------------------------------
# Eval image model
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Eval text model
# ------------------------------------------------------------------------------
roberta_model_name = 'Text_roBERTa_10K_V2'
bert_model_name = 'Text_BERT_10K_V3'

title_model_roberta = build_title_model(2, False)
title_model_roberta.load_weights(f"models/{roberta_model_name}.hdf5")

title_model_bert = build_title_model(2, True)
title_model_bert.load_weights(f"models/{bert_model_name}.hdf5")

title_dataset = text_dataset_basic('test')

# title_model_roberta.evaluate(title_dataset)
# title_model_bert.evaluate(title_dataset)

# y_pred = title_model_bert(title_dataset)
# print(y_pred)

# y_pred = title_model_bert.predict(title_dataset.batch(training_batch_size, drop_remainder=True))

# y_preds = []
# for _, (x, y) in enumerate(title_dataset.as_numpy_iterator()):
#     y_preds.append(title_model_bert(x).argmax(axis=-1), y)
# print(y_preds)
# for _, (_, label) in enumerate(title_dataset.as_numpy_iterator()):
#     print(label)

# df = pd.read_csv(dataset_test_path, sep='\t', header=0)
# y_true = df['2_way_label'].values
# print(y_pred, y_true)

# tb_bert_roberta = np.array(mcnemar_table(y_test,
#                    y_model1=output_bert,
#                    y_model2=output_roberta))


# from mlxtend.evaluate import mcnemar

# chi2, p = mcnemar(ary=tb_bert_roberta, corrected=True)
# print('chi-squared:', chi2)
# print('p-value:', p)



# ------------------------------------------------------------------------------
# Eval dual model
# ------------------------------------------------------------------------------

# score = dual_model.evaluate(test_seq)
