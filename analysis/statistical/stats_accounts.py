import pandas as pd
import dask.dataframe as dd
import math
import time
import numpy as np

# 전체 출력 설정
np.set_printoptions(threshold=np.inf)


file_path = '../../_datasets/account_dataset.parquet'
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
print(f'##### total accounts: {len(df)}')
print(f"##### num of account_id (unique) : {df['account_id'].nunique()}")
print(f"##### age (min/max) : {df['age'].min()}/{df['age'].max()}")
print(f"##### initial_balance (min/max) : {df['initial_balance'].min()}/{df['initial_balance'].max()}")
print(f"##### assigned_bank_type (unique) : {df['assigned_bank_type'].unique()}")
print(f"##### assigned_bank (unique) : {df['assigned_bank'].unique()}")
