import sqlite3

import dask.dataframe as dd
import pandas as pd

db_file = "../db_sqlite/bis.db"
table_name = "laundering"

df_train_target_dataset = pd.read_parquet("../train_target_dataset.parquet")
conn = sqlite3.connect(db_file)
df_train_target_dataset.to_sql(table_name, conn, if_exists="replace", index=False)

print(f"Complete to save db")

df_transaction_dataset = dd.read_parquet("../transaction_dataset.parquet")
merged = df_transaction_dataset.merge(df_train_target_dataset, on=["transaction_id", "account_id"], how="inner")
df_laundering_detail = merged.compute()
df_laundering_detail.to_sql("laundering_detail", conn, if_exists="replace", index=False)

print("Complete to process join with transaction_dataset")

df_account_dataset = dd.read_parquet("../account_dataset.parquet")
merged = df_account_dataset.merge(df_laundering_detail, on=["account_id"], how="right")
df_laundering_full = merged.compute()
df_laundering_full.to_sql("laundering_full", conn, if_exists="replace", index=False)

print("Complete to process join with account_dataset")

df_laundering_full.to_csv("../laundering.csv", index=False)

print("Save laundering.csv")

conn.close()
