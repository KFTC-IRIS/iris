import pandas as pd

# 파일 로드
file_path = "../combined_dataset_sample.csv"
df = pd.read_csv(file_path)

# Timestamp 컬럼 생성 (YYYY-MM-DD HH:MM:SS 형식)
df['timestamp'] = pd.to_datetime(
    "2024-" + df['month'].astype(str) + "-" + df['day'].astype(str) + " " +
    df['hour'].astype(str) + ":" + df['min'].astype(str) + ":" + df['sec'].astype(str),
    format='%Y-%m-%d %H:%M:%S'
)

# 기존 컬럼 삭제
df.drop(columns=['month', 'day', 'hour', 'min', 'sec'], inplace=True)

# 컬럼 순서 조정
column_order = ['timestamp'] + [col for col in df.columns if col not in ['timestamp']]
df = df[column_order]

print(df)

# 결과 저장
output_file = "../combined_dataset_sample_timestamp.csv"
df.to_csv(output_file, index=False)
print(f"Updated file saved to {output_file}")
