import os
import time

import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# 전체 출력 설정
np.set_printoptions(threshold=np.inf)

start = time.time()
# file_path = '../../_datasets/combined_dataset_v0.11.parquet'
file_path = '../../_datasets/combined_dataset_v0.12.parquet'
# ddf = dd.read_parquet(file_path)
# df = ddf.compute()
df = pd.read_parquet(file_path)
end = time.time()
print(f"complete to read parquet file {file_path}, {end - start:.5f} sec")
print(df.head())

# source_col = 'source_deg'
# target_col = 'target_deg'
source_col = 'source_between'
target_col = 'target_between'

stats_source_col = df.groupby('laundering_yn')[source_col].describe()
print(f'{source_col} statistics:')
print(stats_source_col)
stats_source_col.to_csv(f'../_results/graph_cen_hist/{source_col}_stats.csv')

stats_target_col = df.groupby('laundering_yn')[target_col].describe()
print(f'{target_col} statistics:')
print(stats_target_col)
stats_target_col.to_csv(f'../_results/graph_cen_hist/{target_col}_stats.csv')

# log scale
# df['log1p_source_deg'] = np.log1p(df['source_deg'])
# df['log1p_target_deg'] = np.log1p(df['target_deg'])
df[f'log_{source_col}'] = np.log(df[source_col])
df[f'log_{target_col}'] = np.log(df[target_col])

output_dir = "../_results/graph_cen_hist"
for column in [f'log_{source_col}', f'log_{target_col}']:
    start = time.time()
    # laundering_yn=0
    sns.kdeplot(
        df[df['laundering_yn'] == 0][column],
        color='blue',
        label='laundering_yn=0',
        fill=True,
        alpha=0.5
    )
    # laundering_yn=1
    sns.kdeplot(
        df[df['laundering_yn'] == 1][column],
        color='red',
        label='laundering_yn=1',
        fill=True,
        alpha=0.5
    )
    plt.ylabel("Density")
    plt.xticks(rotation=90, ha='right', fontsize=7)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.legend()

    output_path = os.path.join(output_dir, f"{column}_dist.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    end = time.time()
    print(f"Saved combined plot for {column} to {output_path}, {end - start:.5f} sec")
