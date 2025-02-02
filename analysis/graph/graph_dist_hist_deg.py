import os
import time

import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 전체 출력 설정
np.set_printoptions(threshold=np.inf)

start = time.time()
file_path = '../../_datasets/combined_dataset_v0.3.parquet'
ddf = dd.read_parquet(file_path)
df = ddf.compute()
end = time.time()
print(f"complete to read parquet file {file_path}, {end - start:.5f} sec")
print(df.head())

stats_source_deg = df.groupby('laundering_yn')['source_deg'].describe()
print('source_deg statistics:')
print(stats_source_deg)
stats_source_deg.to_csv('../_results/graph_deg_hist/source_deg_stats.csv')

stats_target_deg = df.groupby('laundering_yn')['target_deg'].describe()
print('target_deg statistics:')
print(stats_target_deg)
stats_target_deg.to_csv('../_results/graph_deg_hist/target_deg_stats.csv')

# log scale
# df['log1p_source_deg'] = np.log1p(df['source_deg'])
# df['log1p_target_deg'] = np.log1p(df['target_deg'])
df['log_source_deg'] = np.log(df['source_deg'])
df['log_target_deg'] = np.log(df['target_deg'])

output_dir = "../_results/graph_deg_hist"
for column in ['log_source_deg', 'log_target_deg']:
# for column in ['source_deg', 'target_deg']:
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
