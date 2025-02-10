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
file_path = '../../_datasets/combined_dataset_v0.2.parquet'
# ddf = dd.read_parquet(file_path)
# df = ddf.compute()
df = pd.read_parquet(file_path)
end = time.time()
print(f"complete to read parquet file {file_path}, {end - start:.5f} sec")
print(df.head())

graph_method_name = 'laundering_score' # between, laundering_score
source_col = f'source_{graph_method_name}'
target_col = f'target_{graph_method_name}'

stats_source_col = df.groupby('laundering_yn')[source_col].describe()
print(f'{source_col} statistics:')
print(stats_source_col)
stats_source_col.to_csv(f'../_results/graph_dist_hist/{source_col}_stats.csv')

stats_target_col = df.groupby('laundering_yn')[target_col].describe()
print(f'{target_col} statistics:')
print(stats_target_col)
stats_target_col.to_csv(f'../_results/graph_dist_hist/{target_col}_stats.csv')

# log scale
use_log_scale = False

if use_log_scale:
    target_columns = [f'log_{source_col}', f'log_{target_col}']
    df[target_columns[0]] = np.log(df[source_col])
    df[target_columns[1]] = np.log(df[target_col])
else:
    target_columns = [source_col, target_col]

output_dir = "../_results/graph_dist_hist"
for column in target_columns:
    start = time.time()

    # total
    sns.kdeplot(
        df[column],
        color="blue",
        label='total',
        fill=True,
        alpha=0.5
    )
    # laundering_yn=0
    sns.kdeplot(
        df[df['laundering_yn'] == 0][column],
        color='red',
        label='laundering_yn=0',
        fill=True,
        alpha=0.5
    )
    plt.ylabel("Density")
    plt.xticks(rotation=90, ha='right', fontsize=7)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.legend()

    output_path = os.path.join(output_dir, f"{column}_dist_hist.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    end = time.time()
    print(f"Saved combined plot for {column} to {output_path}, {end - start:.5f} sec")
