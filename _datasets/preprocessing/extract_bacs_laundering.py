import time
import dask.dataframe as dd
import json
from IPython.display import display
import pandas as pd

start = time.time()
file_path = '../combined_dataset_v0.2.parquet'
# 1119.83808 sec
# ddf = dd.read_parquet(file_path)
# df = ddf.compute()
# 198.71684 sec
df = pd.read_parquet(file_path)

end = time.time()
print(f"complete to read parquet file {file_path}, {end - start:.5f} sec")
print(df.head())

display(df.info(verbose=True, memory_usage='deep'))
display(df.memory_usage())

# BACS 및 laundering_yn == 1 필터링
df_filtered = df[(df['payment_system'] == 'BACS') & (df['laundering_yn'] == 1)]
print(df_filtered)
print(len(df_filtered))

df_filtered.to_csv('../bacs_laundering_transactions.csv')
