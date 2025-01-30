import pandas as pd
import numpy as np

file_path = "../../_datasets/combined_dataset.parquet"
df = pd.read_parquet(file_path)
output_file = "../_results/stats_dist_cnt/amount_counts_detailed.csv"

bins = [0, 1000, 5000, 10000, 20000, 25000, df['amount'].max()]
labels = ['0-1K', '1K-5K', '5K-10K', '10K-20K', '20K-25K', f"25K+ ({bins[-1]:.2f})"]

# 구간화
df['amount_binned'] = pd.cut(df['amount'], bins=bins, labels=labels, include_lowest=True)

# laundering_yn 값별 건수
counts = df.groupby(['amount_binned', 'laundering_yn']).size().unstack(fill_value=0)
counts.columns = ['laundering_yn_0', 'laundering_yn_1']

# 인덱스 재설정
counts.reset_index(inplace=True)

# 결과를 CSV로 저장
counts.to_csv(output_file, index=False)
print(f"Saved custom bin counts to {output_file}")
