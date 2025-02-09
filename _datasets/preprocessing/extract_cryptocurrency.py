import time

import dask.dataframe as dd
from IPython.display import display

start = time.time()
file_path = '../combined_dataset_v0.1.parquet'
ddf = dd.read_parquet(file_path)
df = ddf.compute()
# df = pd.read_parquet(file_path)

end = time.time()
print(f"complete to read parquet file {file_path}, {end - start:.5f} sec")
print(df.head())

display(df.info(verbose=True, memory_usage='deep'))
display(df.memory_usage())

# category_1 == cryptocurrency 필터링
df_filtered = df[df['category_1'] == 'cryptocurrency']
print(df_filtered)
print(len(df_filtered))

df_filtered.to_csv('../cryptocurrency_transactions.csv')
