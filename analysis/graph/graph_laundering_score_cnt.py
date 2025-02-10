import pandas as pd
import numpy as np

file_path = "../../_datasets/combined_dataset_v0.2.parquet"
df = pd.read_parquet(file_path)

# [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0)
bins = np.arange(0, 1.1, 0.1)

# source_laundering_score 카운팅
source_counts = pd.cut(df["source_laundering_score"], bins=bins, include_lowest=True, right=False).value_counts().sort_index()

# target_laundering_score 카운팅
target_counts = pd.cut(df["target_laundering_score"], bins=bins, include_lowest=True, right=False).value_counts().sort_index()

# [1.0] 구간 추가
source_counts.loc["1.0"] = (df["source_laundering_score"] == 1.0).sum()
target_counts.loc["1.0"] = (df["target_laundering_score"] == 1.0).sum()

print("Source Laundering Score Distribution:")
print(source_counts)
print("\nTarget Laundering Score Distribution:")
print(target_counts)

labels = [f"[{round(b, 1)}, {round(b + 0.1, 1)})" for b in bins[:-1]] + ["1.0"]
distribution_df = pd.DataFrame({"Range": labels, "Source_Count": source_counts, "Target_Count": target_counts})
output_file = "../_results/graph_dist_hist/laundering_score_dist.csv"
distribution_df.to_csv(output_file, index=False)
