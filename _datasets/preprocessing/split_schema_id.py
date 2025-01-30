import pandas as pd
import os

file_path = "../combined_dataset.parquet"
df = pd.read_parquet(file_path)

output_dir = "../laundering_schema_id"

# laundering_schema_id 값이 있는 transaction만 필터링
filtered_df = df[df['laundering_schema_id'].notnull()]

# laundering_schema_id별 저장
for schema_id, group in filtered_df.groupby('laundering_schema_id'):
    output_file = os.path.join(output_dir, f"transaction_{schema_id}.csv")
    group.to_csv(output_file, index=False)
    print(f"Saved file for {schema_id} to {output_file}")
