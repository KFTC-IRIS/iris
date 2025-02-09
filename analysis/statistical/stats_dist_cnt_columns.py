import pandas as pd

categories = [
    # 'month', 'day', 'hour',
    # 'weekday', 'channel', 'payment_system',
    # 'category_0', 'category_1', 'category_2',
    # 'age',
    # 'assigned_bank_type', 'assigned_bank'
    'source_prop_yn', 'target_prop_yn', 'laundering_yn'
]

file_path = "../../_datasets/combined_dataset_v0.2.parquet"
df = pd.read_parquet(file_path, columns=categories)

# 각 컬럼에 대해 건수 계산 및 저장
for category in categories:
    # laundering_yn 값별 건수 계산
    counts = df.groupby([category, 'laundering_yn']).size().unstack(fill_value=0)

    # 컬럼 이름 변경
    counts.columns = ['laundering_yn_0', 'laundering_yn_1']

    # CSV 파일로 저장
    output_file = f"../_results/stats_dist_cnt/{category}_counts.csv"
    counts.to_csv(output_file)

    print(f"Saved laundering counts by {category} to {output_file}")

