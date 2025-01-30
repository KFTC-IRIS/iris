import os
import sys

import pandas as pd
from imblearn.metrics import classification_report_imbalanced

from sklearn.metrics import roc_auc_score, average_precision_score
import dask.dataframe as dd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# XGBoost
import xgboost as xgb
import time

# 2025.01.31 : snu server
# UserWarning: [00:11:25] WARNING: /workspace/src/context.cc:196: XGBoost is not compiled with CUDA support.

start = time.time()
# df = pd.read_csv(f'./_datasets/combined_dataset_sample.csv')
df = pd.read_parquet('../_datasets/combined_dataset.parquet')
# ddf = dd.read_parquet('./_datasets/combined_dataset.parquet')
# df = ddf.compute()
end = time.time()
print(f'##### read datasets: {end - start:.5f} sec')

df.drop(columns=['laundering_schema_type', 'laundering_schema_id'], inplace=True)
print(f'final columns: {df.columns}')

df = df.astype({
    'weekday': 'category',
    'transaction_direction': 'category',
    'channel': 'category',
    'payment_system': 'category',
    'category_0': 'category',
    'category_1': 'category',
    'category_2': 'category',
    'age': 'float',
    'counterpart_id': 'category',
    'assigned_bank_type': 'category',
    'assigned_bank': 'category'
})

print(df.dtypes)


df_train, df_eval = train_test_split(df, test_size=0.3, shuffle=False)
df_valid, df_test = train_test_split(df_eval, test_size=0.5, shuffle=False)

# train dataset
train_labels = df_train['laundering_yn']
le = LabelEncoder()
df_train.drop(columns=['laundering_yn'], inplace=True)

valid_labels = df_valid['laundering_yn']
df_valid.drop(columns=['laundering_yn'], inplace=True)

# test dataset
test_labels = df_test['laundering_yn']
df_test.drop(columns=['laundering_yn'], inplace=True)

model = xgb.XGBClassifier(
    tree_method = "hist",
    device = "cuda",
    n_estimators=10,  # 1000
    learning_rate=1e-3,
    max_depth=10,
    gamma=0.0,
    reg_lambda=1.0,
    enable_categorical=True,
    max_cat_to_onehot=1,
    n_jobs=8,
    random_state=2025
)

start = time.time()
model.fit(df_train, train_labels, eval_set=[(df_valid, valid_labels)])
end = time.time()
print(f'##### fit model by using datasets: {end - start:.5f} sec')

start = time.time()
pred_labels = model.predict(df_test)
end = time.time()
print(f'##### predict: {end - start:.5f} sec')

im_metrics = classification_report_imbalanced(test_labels, pred_labels, digits=5, output_dict=True)

print('##### imbalanced metrics')
print(im_metrics)

auc, ap = None, None
try:
    auc = roc_auc_score(test_labels, pred_labels)
    ap = average_precision_score(test_labels, pred_labels)
except:
    print('roc-auc or ap metric exception!')
print(f'##### auc: {auc}, ap: {ap}')
