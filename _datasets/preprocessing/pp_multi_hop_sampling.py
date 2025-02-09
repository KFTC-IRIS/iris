import pandas as pd
import os
import time

file_path = "../combined_dataset_v0.1.parquet"
start = time.time()
df = pd.read_parquet(file_path)
end = time.time()
print(f"##### complete to read {file_path} : {end - start:.5f} sec")

# 0-hop: 자금세탁거래 필터링 (laundering_yn == 1)
df_h0 = df[df['laundering_yn'] == 1].copy()
print(len(df_h0))

# 자금세탁거래에 포함된 source/target 추출 (set으로 중복 제거)
ml_nodes_h0 = set(df_h0['source']).union(set(df_h0['target']))
df_h0.to_parquet(f'../combined_0_hop_datasets.parquet', index=False)
print(f"##### 0-hop sampling : # transactions = {len(df_h0)} (unique: {df_h0['transaction_id'].nunique()}), # nodes = {len(ml_nodes_h0)}")

k = 1
current_nodes = ml_nodes_h0.copy()

for hop in range(1, k + 1):
    # 현재까지 발견된 노드들과 연결된 거래 찾기
    df_hop = df[df['source'].isin(current_nodes) | df['target'].isin(current_nodes)].copy()

    # 노드 리스트 업데이트
    new_nodes = set(df_hop['source']).union(set(df_hop['target']))

    # 이전 노드 와의 차이를 계산하여 신규 노드만 추가
    new_nodes -= current_nodes
    current_nodes.update(new_nodes)

    print(f"##### {hop}-hop sampling : # transactions = {len(df_hop)} (unique: {df_hop['transaction_id'].nunique()}), # nodes = {len(current_nodes)}")
    df_hop.to_parquet(f'../combined_{hop}_hop_datasets.parquet', index=False)
