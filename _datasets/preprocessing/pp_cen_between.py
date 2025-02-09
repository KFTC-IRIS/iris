import networkx as nx
import numpy as np
import pandas as pd
import dask.dataframe as dd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import nx_cugraph as nxcg
import pickle
import time
import dask.dataframe as dd
import json
import cudf
import cugraph

def create_graph(df: pd.DataFrame) -> nx.DiGraph:
    # transaction_id,
    # account_id,
    # month,day,weekday,hour,min,sec,
    # transaction_direction,
    # channel,payment_system,category_0,category_1,category_2,
    # amount,
    # counterpart_id,
    # age,
    # initial_balance,
    # assigned_bank_type,
    # assigned_bank,laundering_schema_type,laundering_schema_id,laundering_yn
    return nx.from_pandas_edgelist(
        df,
        source='source',
        target='target',
        edge_attr=['transaction_id', 'amount'],
        edge_key='transaction_id',
        create_using=nx.MultiDiGraph()
    )

if __name__ == "__main__":

    start = time.time()
    file_path = '../combined_dataset_v0.1.parquet'
    ddf = dd.read_parquet(file_path)
    df = ddf.compute()
    # df = pd.read_parquet(file_path)
    end = time.time()
    print(f"complete to read parquet file {file_path}, {end - start:.5f} sec")
    print(df.head())

    between_cen_file = '../between_centrality_cd_v0.1.json'
    if os.path.exists(between_cen_file):
        with open(between_cen_file, 'r') as f:
            between_cen = json.load(f)  # JSON 파일에서 로드
            between_cen = {k: float(v) for k, v in between_cen.items()}  # float 변환

        start = time.time()
        df['source_between'] = df['source'].map(between_cen).fillna(0)
        df['target_between'] = df['target'].map(between_cen).fillna(0)
        end = time.time()
        print(f"complete to create source/target between column {end - start:.5f} sec")

        start = time.time()
        output_file = "../combined_dataset_v0.12.parquet"
        df.to_parquet(output_file, index=False)
        end = time.time()
        print(f"complete to save {output_file}, {end - start:.5f} sec")
    else:
        start = time.time()
        G = create_graph(df)
        end = time.time()
        print(f'complete to construct graph {end - start:.5f} sec')

        start = time.time()
        G_cg = nxcg.from_networkx(G)
        end = time.time()
        print(f'convert networkx to cugraph {end - start:.5f} sec')

        start = time.time()
        # degree_cen = nx.degree_centrality(G_cg, backend="cugraph")
        # close_cen = nx.closeness_centrality(G)
        # between_cen = nx.betweenness_centrality(G, k=1000)
        between_cen = nx.betweenness_centrality(G_cg, k=1000, backend="cugraph")
        end = time.time()
        print(f'complete to calculate between centrality {end - start:.5f} sec')

        start = time.time()
        with open(between_cen_file, 'w') as f:
            json.dump(between_cen, f)  # JSON 형식으로 저장
        end = time.time()
        print(f"Saved between centrality in {end - start:.5f} sec")
