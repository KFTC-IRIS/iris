from typing import Dict, Any

import dask.dataframe as dd
import networkx as nx
import nx_cugraph as nxcg
import pandas as pd
import time
import os
import community

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
        source='account_id',
        target='counterpart_id',
        edge_attr=['transaction_id', 'month', 'day', 'hour', 'min', 'sec', 'amount'],
        edge_key='transaction_id',
        create_using=nx.MultiDiGraph()
    )

def calculate_graph_features(graph: nx.DiGraph) -> Dict[str, Any]:
    results = {
        'number_of_nodes': graph.number_of_nodes(),
        'number_of_edges': graph.number_of_edges(),
        'density': nx.density(graph),
        # 'num_of_scc': nx.number_strongly_connected_components(graph, backend='networkx')
    }
    return results


if __name__ == "__main__":
    graph_path = '../_datasets/combined_G.pkl'
    # graph_path = '../_datasets/combined_G.graphml'
    if os.path.isfile(graph_path):
        start = time.time()
        # G = nx.read_gml(path=graph_path)
        G = nx.read_gpickle(graph_path)
        # G = read_graphml(graph_path)
        end = time.time()
        print(f'##### read graph file : {end - start:.5f} sec')
    else:
        file_path = '../_datasets/combined_dataset.parquet'
        ddf = dd.read_parquet(file_path)
        df = ddf.compute()
        print(df.head())

        # 그래프 생성
        start = time.time()
        G = create_graph(df)
        end = time.time()
        print(f'##### construct graph: {end - start:.5f} sec')
        # nx.write_gml(G, path=graph_path)  # issue : 62GB
        nx.write_gpickle(G, graph_path)
        # nx.write_graphml_lxml(G, graph_path)

    start = time.time()
    G_cg = nxcg.from_networkx(G)
    end = time.time()
    print(f'##### convert networkx to cugraph: {end - start:.5f} sec')

    # start = time.time()
    # G_feat = calculate_graph_features(G_cg)
    # end = time.time()
    # print(f'##### calculate graph features: {end - start:.5f} sec')
    # print(G_feat)

    # start = time.time()
    # deg_cen = nx.degree_centrality(G_cg, backend="cugraph")
    # end = time.time()
    # print(f'##### calculate degree centrality: {end - start:.5f} sec')

    # start = time.time()
    # clo_cen = nx.closeness_centrality(G_cg, backend="networkx")
    # end = time.time()
    # print(f'##### calculate closeness centrality: {end - start:.5f} sec')

    # start = time.time()
    # bet_cen = nx.betweenness_centrality(G_cg, k=100, backend="cugraph")
    # end = time.time()
    # print(f'##### complete betweenness_centrality: {end - start:.5f} sec')

    start = time.time()
    # partition = nx.community.louvain_communities(G_cg, backend="networkx")
    partition = community.best_partition(G_cg)
    print('##### Community Detection (Louvain)')
    print(f'Number of communities: {len(set(partition.values()))}')
    end = time.time()
    print(f'##### complete louvain_communities: {end - start:.5f} sec')
