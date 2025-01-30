import pandas as pd
import numpy as np

file_path = "../../_datasets/combined_dataset.parquet"
df = pd.read_parquet(file_path)
output_file = "../_results/stats_dist_cnt/initial_balance_counts_detailed.csv"

bins = [-1000, 0, 10000, 100000, 1000000, 2000000, df['initial_balance'].max()]
labels = ['-1K-0', '0-10K', '10K-100K', '100K-1000K', '1000K-2000K', f"2000K+ ({bins[-1]:.2f})"]

# 구간화
df['balance_binned'] = pd.cut(df['initial_balance'], bins=bins, labels=labels, include_lowest=True)

# laundering_yn 값별 건수
counts = df.groupby(['balance_binned', 'laundering_yn']).size().unstack(fill_value=0)
counts.columns = ['laundering_yn_0', 'laundering_yn_1']

# 인덱스 재설정
counts.reset_index(inplace=True)

# 결과를 CSV로 저장
counts.to_csv(output_file, index=False)
print(f"Saved custom bin counts to {output_file}")
