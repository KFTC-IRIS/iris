import pandas as pd
import dask.dataframe as dd

# 데이터 로드
file_path = "../combined_dataset_v0.1.parquet"
ddf = dd.read_parquet(file_path)
df = ddf.compute()
# df = pd.read_parquet(file_path)
output_path = "../combined_dataset_v0.1_sample.csv"

# 샘플링할 전체 거래 수 및 비율 설정
n_samples = 10000  # 샘플링할 전체 거래 수
laundering_ratio = 0.10  # 자금세탁 거래 비율

# 이상거래(비정상 거래) 샘플링 개수 계산
n_laundering_samples = int(n_samples * laundering_ratio)

# laundering_yn=1 샘플링
df_fraud_sample = df[df["laundering_yn"] == 1].sample(
    n=min(n_laundering_samples, len(df[df["laundering_yn"] == 1])),
    random_state=42
)

# laundering_yn=0 샘플링
df_normal_sample = df[df["laundering_yn"] == 0].sample(
    n=min(n_samples - n_laundering_samples, len(df[df["laundering_yn"] == 0])),
    random_state=42
)

df_sample = pd.concat([df_fraud_sample, df_normal_sample])
df_sample.to_csv(output_path, index=False)
print(f"📁 Sampled data saved to: {output_path}")
