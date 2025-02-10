import community as community_louvain
import dask.dataframe as dd
import json
import networkx as nx
import os
import pandas as pd
import random
import time
from collections import Counter
import numpy as np

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
        # create_using=nx.MultiDiGraph()
        create_using=nx.Graph()
    )


def propagation_laundering_score(G, df, max_iter=10):
    # community detection (community shape: {node: community id})
    start = time.time()
    communities = community_louvain.best_partition(G)
    end = time.time()
    print(f"initial louvain communities : {len(set(communities.values()))}, {end - start:.5f} sec")

    start = time.time()
    with open('../communities_cd_v0.1.json', 'w') as f:
        json.dump(communities, f)  # JSON 형식으로 저장
    end = time.time()
    print(f"save louvain communities, {end - start:.5f} sec")

    # 벡터 연산 기반으로 초기 laundering_score 설정
    start = time.time()
    laundering_score = {}
    source_nodes = df["source"].values
    target_nodes = df["target"].values
    labels = df["laundering_yn"].values

    # 자금세탁거래에 포함된 노드(source/target)는 1로 확정
    laundering_score.update({s: 1 for s, l in zip(source_nodes, labels) if l == 1})
    laundering_score.update({t: 1 for t, l in zip(target_nodes, labels) if l == 1})
    # 자금세탁 이외 거래들에 포함된 노드는 0으로 설정 후 업데이트
    laundering_score.update({s: 0 for s in source_nodes if s not in laundering_score})
    laundering_score.update({t: 0 for t in target_nodes if t not in laundering_score})
    end = time.time()
    print(f"initialize laundering score {end - start:.5f} sec")

    # propagation (semi-supervised)
    neighbors_dict = {str(node): [str(nbr) for nbr in G.neighbors(node)] for node in G.nodes()}
    for i in range(max_iter):
        start = time.time()

        nodes = np.array(list(G.nodes()))
        np.random.shuffle(nodes)  # NumPy 기반 랜덤 셔플

        new_scores = {}
        for node in nodes:
            print(f'process {i + 1} iter, node: {node} ...')
            if laundering_score[node] == 1:
                continue  # 자금세탁 거래에 포함된 노드는 제외

            neighbors = neighbors_dict[node]
            if not neighbors:
                continue  # 이웃이 없으면 제외

            # 벡터 연산으로 평균값 계산
            neighbor_scores = np.array([laundering_score[n] for n in neighbors if n in laundering_score])
            if len(neighbor_scores) == 0:
                continue
            new_scores[node] = np.mean(neighbor_scores)

        laundering_score.update(new_scores)

        end = time.time()
        print(f'complete {i + 1} iter, {end - start:.5f} sec')

    return laundering_score


if __name__ == "__main__":

    start = time.time()
    file_path = '../combined_dataset_v0.1.parquet'
    # file_path = '../combined_dataset_v0.1_sample.csv'
    ddf = dd.read_parquet(file_path)
    df = ddf.compute()
    # df = pd.read_csv(file_path)
    end = time.time()
    print(f"complete to read parquet file {file_path}, {end - start:.5f} sec")
    print(df.head())
    print(f"Original Data Laundering Count:\n{df['laundering_yn'].value_counts()}")
    # source와 target의 유니크 값 개수 출력
    print(
        f"unique count source: {df['source'].nunique()}, target: {df['target'].nunique()}, total : {pd.concat([df['source'], df['target']]).nunique()}")

    label_propagation_file = '../laundering_score_cd_v0.1.json'
    # label_propagation_file = '../laundering_score_cd_v0.1_sample.json'
    if os.path.exists(label_propagation_file):
        print('add source/target laundering score column')
        with open(label_propagation_file, 'r') as f:
            laundering_score = json.load(f)
            laundering_score = {k: float(v) for k, v in laundering_score.items()}

        start = time.time()
        df['source_laundering_score'] = df['source'].map(laundering_score).fillna(0)
        df['target_laundering_score'] = df['target'].map(laundering_score).fillna(0)
        end = time.time()
        print(f"complete to create source/target laundering score column {end - start:.5f} sec")

        start = time.time()
        output_file = "../combined_dataset_v0.13.parquet"
        df.to_parquet(output_file, index=False)
        # output_file = "../combined_dataset_v0.13_sample.csv"
        # df.to_csv(output_file, index=False)
        end = time.time()
        print(f"complete to save {output_file}, {end - start:.5f} sec")
    else:
        print('construct graph ...')
        start = time.time()
        G = create_graph(df)
        end = time.time()
        print(f'complete to construct graph {end - start:.5f} sec')

        start = time.time()
        """
        - networkx label_propagation 사용 시 전체 데이터의 경우 모든 label이 0으로 설정되는 문제 발생
          - label_propagation_communities (무방향)
          - fast_label_propagation_communities (방향)
        - 따라서 louvain 알고리즘(무방향)을 사용해 별도 label propagation을 구현
          - louvain 알고리즘을 그대로 사용할 경우 community detection 결과(커뮤니티 번호)만 생성되기 때문에,
          - 자금세탁거래에 참여한 노드를 사용해 label propagation 수행  
        """
        # communities = label_propagation.fast_label_propagation_communities(G)
        # communities = community_louvain.best_partition(G)
        laundering_score = propagation_laundering_score(G, df)
        print(min(laundering_score.values()), max(laundering_score.values()))

        end = time.time()
        print(f'complete to calculate propagation score {end - start:.5f} sec')

        start = time.time()
        with open(label_propagation_file, 'w') as f:
            json.dump(laundering_score, f)  # JSON 형식으로 저장
        end = time.time()
        print(f"Saved propagation score in {end - start:.5f} sec")
