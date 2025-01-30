import pandas as pd
import dask.dataframe as dd
import math
import time
import numpy as np

import os

import cupy as cp
import dask_cudf
import pandas as pd

import cudf

# 전체 출력 설정
np.set_printoptions(threshold=np.inf)

# file_path = './_datasets/transaction_dataset.parquet'
combine_file_path = '../_datasets/combined_dataset.parquet'
start = time.time()

# 155 서버 : 33.73600 sec
# runpod a6000 : 113.22369 sec
# df = pd.read_parquet(combine_file_path)
# print(df.head())
# print(len(df))


# 155 서버 : 39.46279 sec
# runpod a6000 :
# snu : 112.97518 sec

ddf = dd.read_parquet(combine_file_path)
df = ddf.compute()
print(df.head())

# 155 서버
# NotImplementedError: dictionary<values=string, indices=int32, ordered=0>
# df = dask_cudf.read_parquet(combine_file_path)
# print(df.head())

end = time.time()
print(f"{end - start:.5f} sec")
