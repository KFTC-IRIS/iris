import pandas as pd
import dask.dataframe as dd

file_path = '../combined_dataset_v0.1.parquet'
ddf = dd.read_parquet(file_path)
df = ddf.compute()
print(f"complete to read parquet file {file_path}")

# 벡터 연산을 사용한 source 및 target 컬럼 생성
df['source'] = df['account_id'].astype(str)
df['target'] = df['counterpart_id'].astype(str)

# 조건에 맞게 값을 변경 (벡터 연산 적용)
mask_inbound = df['transaction_direction'] == 'inbound'

df.loc[mask_inbound, 'source'] = df.loc[mask_inbound, 'counterpart_id'].astype(str)
df.loc[mask_inbound, 'target'] = df.loc[mask_inbound, 'account_id'].astype(str)

# 컬럼 순서 조정
column_order = ['transaction_id', 'source', 'target'] + [col for col in df.columns if col not in ['transaction_id', 'source', 'target']]
df = df[column_order]

# 중복 컬럼 제거
df.drop(['account_id', 'transaction_direction', 'counterpart_id'], inplace=True, axis=1)

output_file = "../combined_dataset_st.parquet"
df.to_parquet(output_file, index=False)
print(f"Updated file saved to {output_file}")
