import pandas as pd
import dask.dataframe as dd
import math
import time
import numpy as np

# 전체 출력 설정
np.set_printoptions(threshold=np.inf)

file_path = '../../_datasets/transaction_dataset.parquet'
start = time.time()

# 28.99s
df = pd.read_parquet(file_path)
print(df.head())

# 14.36s
# ddf = dd.read_parquet(file_path)
# df = ddf.compute()
# print(df.head())

end = time.time()
print(f"{end - start:.5f} sec")

print(df.columns.to_list())
print(f'##### total transaction: {len(df)}')


# print(f"##### transaction_id (unique) : {df['transaction_id'].nunique()}")
# print(f"##### account_id (unique) : {df['account_id'].nunique()}")
# print(f"##### month (unique) : {df['month'].unique()}")
# print(f"##### day (unique) : {df['day'].unique()}")
# print(f"##### weekday (unique) : {df['weekday'].unique()}")
# print(f"##### hour (unique) : {df['hour'].unique()}")
# print(f"##### min (unique) : {df['min'].unique()}")
# print(f"##### sec (unique) : {df['sec'].unique()}")
# print(f"##### transaction_direction (unique) : {df['transaction_direction'].unique()}")
# print(f"##### channel (unique) : {df['channel'].unique()}")
# print(f"##### payment_system (unique) : {df['payment_system'].unique()}")
# print(f"##### category_0 (unique) : {df['category_0'].unique()}")
# print(f"##### category_1 (unique) : {df['category_1'].unique()}")
# print(f"##### category_2 (unique) : {df['category_2'].unique()}")
# print(f"##### amount (min/max) : {df['amount'].min()}/{df['amount'].max()}")
# print(f"##### counterpart_id (unique) : {df['counterpart_id'].unique()}")

# # counterpart_id 중 account_id(숫자) row 제거
# non_numeric_counterpart_ids = df['counterpart_id'][~df['counterpart_id'].astype(str).str.isdigit()].unique()
# print(f"##### Total counterpart_id values: {len(df['counterpart_id'])}")
# print(f"##### Non-numeric counterpart_id values ({len(non_numeric_counterpart_ids)}): {non_numeric_counterpart_ids}")
#
# # id 넘버링 제거 (id_00 → id)
# non_numeric_series = pd.Series(non_numeric_counterpart_ids)
# cleaned_ids = non_numeric_series.str.replace(r'_\d+$', '', regex=True)
# unique_cleaned_ids = cleaned_ids.unique()
# print(f"##### Non-numeric counterpart_id values after cleaning ({len(unique_cleaned_ids)}): {unique_cleaned_ids}")

