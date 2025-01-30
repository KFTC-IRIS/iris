import pandas as pd

# 데이터 로드
file_path = "../../_datasets/combined_dataset.parquet"
df = pd.read_parquet(file_path)

# 처리할 컬럼 목록
categories = [
    'month', 'day', 'hour',
    'weekday', 'channel', 'payment_system',
    'category_0', 'category_1', 'category_2',
    'age',
    'assigned_bank_type', 'assigned_bank'
]

# 각 컬럼에 대해 건수 계산 및 저장
for category in categories:
    # laundering_yn 값별 건수 계산
    counts = df.groupby([category, 'laundering_yn']).size().unstack(fill_value=0)

    # 컬럼 이름 변경
    counts.columns = ['laundering_yn_0', 'laundering_yn_1']

    # CSV 파일로 저장
    output_file = f"./_results/stats_cnt/{category}_counts.csv"
    counts.to_csv(output_file)

    print(f"Saved laundering counts by {category} to {output_file}")

