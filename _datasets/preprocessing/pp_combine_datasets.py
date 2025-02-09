import pandas as pd
import numpy as np

# 전체 출력 설정
np.set_printoptions(threshold=np.inf)

# transactions : 307,596,577
transaction_file = "../transaction_dataset.parquet"
df_transaction = pd.read_parquet(transaction_file)
print(f'##### complete to read {transaction_file} : {len(df_transaction)}')

# target : 35,107
train_target_file = "../train_target_dataset.parquet"
df_train_target = pd.read_parquet(train_target_file)
print(f'##### complete to read {train_target_file} : {len(df_train_target)}')

# accounts : 1,809,646
account_file = "../account_dataset.parquet"
df_account = pd.read_parquet(account_file)
print(f'##### complete to read {account_file} : {len(df_account)}')

# transaction + accounts 병합 (rows : 307,596,577)
print("##### merging account data...")
df_transaction = df_transaction.merge(
    df_account[['account_id', 'age', 'initial_balance', 'assigned_bank_type', 'assigned_bank']],
    on='account_id',
    how='left'
)
print(f'##### complete to merge transactions : {len(df_transaction)}')

# transaction + target 병합 (rows : 307,596,577)
print("##### merging train target data...")
df_transaction['account_id'] = df_transaction['account_id'].astype(int)
df_train_target['account_id'] = df_train_target['account_id'].astype(int)
df_transaction = df_transaction.merge(
    df_train_target[['transaction_id', 'account_id', 'laundering_schema_type', 'laundering_schema_id']],
    on=['transaction_id', 'account_id'],
    how='left'
)
print(f'##### complete to merge transactions : {len(df_transaction)}')

# laundering_yn 레이블 추가 (rows : 307,596,577)
print("##### creating laundering_yn column...")
df_transaction['laundering_yn'] = df_transaction[['laundering_schema_type', 'laundering_schema_id']].notnull().all(axis=1).astype(int)
print(f'##### complete to create laundering_yn : {len(df_transaction)}')

# source / target 컬럼 생성 (rows : 307,596,577)
print("##### creating source/target columns...")
df_transaction['source'] = df_transaction['account_id'].astype(str)
df_transaction['target'] = df_transaction['counterpart_id'].astype(str)
mask_inbound = df_transaction['transaction_direction'] == 'inbound'
df_transaction.loc[mask_inbound, 'source'] = df_transaction.loc[mask_inbound, 'counterpart_id'].astype(str)
df_transaction.loc[mask_inbound, 'target'] = df_transaction.loc[mask_inbound, 'account_id'].astype(str)
print(f'##### complete to create source/target columns : {len(df_transaction)}')

# 컬럼 순서 변경 (rows : 307,596,577)
column_order = ['transaction_id', 'source', 'target'] + [col for col in df_transaction.columns if col not in ['transaction_id', 'source', 'target']]
df_transaction = df_transaction[column_order]
# 기존 컬럼 제거
df_transaction.drop(['account_id', 'transaction_direction', 'counterpart_id'], inplace=True, axis=1)
print(f'##### complete to drop account_id/transaction_direction/counterpart_id columns : {len(df_transaction)}')

# 중복 row 계산 : 6.12%
duplicate_ratio = df_transaction.duplicated(subset=['transaction_id']).mean() * 100
print(f"##### duplicate transaction_id ratio: {duplicate_ratio:.2f}%")

# 중복 행 제거 (row 수 : 288,785,789)
df_transaction = df_transaction.drop_duplicates(subset=['transaction_id'], keep='first')
print(f"##### removed duplicate transactions. Remaining rows: {len(df_transaction)}")

output_file = "../combined_dataset_v0.1.parquet"
df_transaction.to_parquet(output_file, index=False)
print(f"Updated file saved to {output_file}")
