import pandas as pd

df = pd.read_parquet("../combined_dataset_v0.1.parquet")
print(f"complete to read v0.1")
df_11 = pd.read_parquet("../combined_dataset_v0.11.parquet", columns=['source_deg', 'target_deg'])
print(f"complete to read v0.11")
df_12 = pd.read_parquet("../combined_dataset_v0.12.parquet", columns=['source_between', 'target_between'])
print(f"complete to read v0.12")
df_13 = pd.read_parquet("../combined_dataset_v0.13.parquet", columns=['source_prop_yn', 'target_prop_yn'])
print(f"complete to read v0.13")

# 데이터 병합
df_combined = pd.concat([df, df_11, df_12, df_13], axis=1)

# 저장 (압축을 적용하여 용량 절감)
df_combined.to_parquet("../combined_dataset_v0.2.parquet")

print(f"complete to save v0.2")
