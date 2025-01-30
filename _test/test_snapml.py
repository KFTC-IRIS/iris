from snapml import GraphFeaturePreprocessor
import numpy as np
import time
import json
import pandas as pd
from IPython.display import display
from datetime import datetime

# 전체 출력 설정
np.set_printoptions(threshold=np.inf)
pd.options.display.max_columns = None

# Path to the file that contains financial transactions, e.g., used for training ML models
train_graph_path = "../_datasets/combined_dataset_sample_a_to_a_100.csv"

print("Loading the transactions ")
df = pd.read_csv(train_graph_path)
# object → int
df['age'] = pd.to_numeric(df['age'], errors='coerce')

# 필요한 컬럼만 선택
required_columns = ['transaction_id', 'month', 'day', 'hour', 'min', 'sec', 'account_id', 'counterpart_id', 'transaction_direction', 'amount']
data = df[required_columns]

# 에지 리스트 생성
edges = []
for index, row in data.iterrows():
    # print(index, row)

    if row['transaction_direction'] == 'outbound':
        source = row['account_id']
        target = row['counterpart_id']
    elif row['transaction_direction'] == 'inbound':
        source = row['counterpart_id']
        target = row['account_id']
    else:
        continue  # 유효하지 않은 방향 처리

    # row['month', row['day'], row['hour'], row['min'], row['sec']
    # timestamp = datetime(
    #     year=2025,  # 기본 연도를 설정
    #     month=int(row['month']),
    #     day=int(row['day']),
    #     hour=int(row['hour']),
    #     minute=int(row['min']),
    #     second=int(row['sec']),
    # ).strftime('%Y%m%d%H%M%S')

    timestamp = index

    # (Edge ID, Source, Target, Timestamp, Additional Features)
    # edges.append([index, source, target, timestamp, row['amount']])
    # edges.append(index, source, target, int(timestamp)])
    edges.append([row['transaction_id'], source, target, int(timestamp), row['amount']])
    # display(edges)

# 에지 리스트를 데이터프레임으로 변환
edges_df = pd.DataFrame(edges, columns=['Edge ID', 'Source Vertex ID', 'Target Vertex ID', 'Timestamp', 'Amount'])
# edges_df = pd.DataFrame(edges, columns=['Edge ID', 'Source Vertex ID', 'Target Vertex ID', 'Timestamp'])
edges_df = edges_df[:7]

edges_df = edges_df.sort_values(by=['Timestamp']).reset_index(drop=True)
print(edges_df)

edges_np = edges_df.to_numpy()

params = {
    "num_threads": 4,  # number of software threads to be used (important for performance)
    # "time_window": 16,  # time window used if no pattern was specified
    "time_window": 16,  # time window used if no pattern was specified

    "vertex_stats": True,  # produce vertex statistics
    # "vertex_stats_cols": [3],  # produce vertex statistics using the selected input columns
    "vertex_stats_cols": [4],  # produce vertex statistics using the selected input columns

    # features: 0:fan,1:deg,2:ratio,3:avg,4:sum,5:min,6:max,7:median,8:var,9:skew,10:kurtosis
    "vertex_stats_feats": [0, 1, 2, 3, 4, 8, 9, 10],  # fan,deg,ratio,avg,sum,var,skew,kurtosis
    # "vertex_stats_feats": [0, 1],  # fan,deg,ratio,avg,sum,var,skew,kurtosis

    # fan in/out parameters
    "fan": True,
    "fan_tw": 16,
    "fan_bins": [y + 2 for y in range(2)],

    # in/out degree parameters
    "degree": True,
    "degree_tw": 16,
    "degree_bins": [y + 2 for y in range(2)],

    # scatter gather parameters
    "scatter-gather": True,
    "scatter-gather_tw": 16,
    "scatter-gather_bins": [y + 2 for y in range(2)],

    # temporal cycle parameters
    "temp-cycle": True,
    "temp-cycle_tw": 16,
    "temp-cycle_bins": [y + 2 for y in range(2)],

    # length-constrained simple cycle parameters
    "lc-cycle": False,
    "lc-cycle_tw": 16,
    "lc-cycle_len": 8,
    "lc-cycle_bins": [y + 2 for y in range(2)],
}

# Create a Graph Feature Preprocessor, set its configuration using the above dictionary and verify it

print("Creating a graph feature preprocessor ")
gp = GraphFeaturePreprocessor()

print("Setting the parameters of the graph feature preprocessor ")
gp.set_params(params)

print("Graph feature preprocessor parameters: ", json.dumps(gp.get_params(), indent=4))

print("Enriching the transactions with new graph features ")
print("Raw dataset shape: ", edges_np.shape)

# the fit_transform and transform functions are equivalent
# these functions can run on single transactions or on batches of transactions
X_train_enriched = gp.fit_transform(edges_np.astype("float64"))
# X_train_enriched = gp.fit(edges_np.astype("float64"))

print("Enriched dataset shape: ", X_train_enriched.shape)

