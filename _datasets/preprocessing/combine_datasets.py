import pandas as pd
import numpy as np

# 전체 출력 설정
np.set_printoptions(threshold=np.inf)

print("Loading datasets...")
transaction_file = "../transaction_dataset.parquet"
transactions = pd.read_parquet(transaction_file)
print(f'##### Step 1 num transactions: {len(transactions)}')

train_target_file = "../train_target_dataset.parquet"
train_target = pd.read_parquet(train_target_file)

account_file = "../account_dataset.parquet"
accounts = pd.read_parquet(account_file)

output_file = "../combined_dataset.parquet"

print("Merging account data...")
transactions = transactions.merge(
    accounts[['account_id', 'age', 'initial_balance', 'assigned_bank_type', 'assigned_bank']],
    on='account_id',
    how='left'
)
print(f'##### Step 2 num transactions: {len(transactions)}')

transactions['account_id'] = transactions['account_id'].astype(int)
train_target['account_id'] = train_target['account_id'].astype(int)

print("Transactions columns:", transactions.columns.tolist())
print("Train target columns:", train_target.columns.tolist())

print("Merging train target data...")
transactions = transactions.merge(
    train_target[['transaction_id', 'account_id', 'laundering_schema_type', 'laundering_schema_id']],
    on=['transaction_id', 'account_id'],
    how='left'
)
print(f'##### Step 3 num transactions: {len(transactions)}')

print("Creating laundering_yn column...")
transactions['laundering_yn'] = transactions[['laundering_schema_type', 'laundering_schema_id']].notnull().all(axis=1).astype(int)
print(f'##### Step 4 num transactions: {len(transactions)}')

print("Saving the combined dataset...")
transactions.to_parquet(output_file, index=False)
print(f"Combined dataset saved to {output_file}")

transactions.to_parquet("../combined_dataset.parquet", index=False)
# transactions[:1000].to_csv("./_datasets/combined_dataset_sample.csv", index=False)

# num_samples = 100
#
# file_path = './_datasets/combined_dataset.parquet'
# df = pd.read_parquet(file_path)
#
# # sampling account to account
# df_filtered = df[pd.to_numeric(df['counterpart_id'], errors='coerce').notnull()]
# df_filtered['counterpart_id'] = pd.to_numeric(df_filtered['counterpart_id'])
# df_sample = df_filtered.head(num_samples)
#
# df_sample.to_csv('./_datasets/combined_dataset_sample_a_to_a_100.csv', index=False)