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

def extract_value_from_dict(row, centralities, column_name):
    key = row[column_name]
    value = centralities.get(key, 0)
    return value

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
    file_path = '../combined_dataset_v0.2.parquet'
    ddf = dd.read_parquet(file_path)
    df = ddf.compute()
    end = time.time()
    print(f"complete to read parquet file {file_path}, {end - start:.5f} sec")
    print(df.head())

    degree_cen_file = '../degree_centrality.json'
    if os.path.exists(degree_cen_file):
        with open(degree_cen_file, 'r') as f:
            degree_cen = json.load(f)  # JSON 파일에서 로드
            degree_cen = {k: float(v) for k, v in degree_cen.items()}  # float 변환

        start = time.time()
        # df['source_deg'] = df.apply(extract_value_from_dict, axis=1,
        #                            args=(degree_cen, 'source'))
        # df['target_deg'] = df.apply(extract_value_from_dict, axis=1,
        #                            args=(degree_cen, 'target'))
        df['source_deg'] = df['source'].map(degree_cen).fillna(0)
        df['target_deg'] = df['target'].map(degree_cen).fillna(0)
        end = time.time()
        print(f"complete to create source/target degree column {end - start:.5f} sec")

        output_file = "../combined_dataset_v0.3.parquet"
        df.to_parquet(output_file, index=False)
        print(f"complete to save {output_file}, {end - start:.5f} sec")
    else:
        start = time.time()
        G = create_graph(df)
        end = time.time()
        print(f'complete to construct graph {end - start:.5f} sec')

        # with open('../../_datasets/combined_G.pkl', 'wb') as f:
        #     pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
        # print('complete to write graph')

        # start = time.time()
        # G_cg = nxcg.from_networkx(G)
        # end = time.time()
        # print(f'convert networkx to cugraph {end - start:.5f} sec')

        start = time.time()
        # degree_cen = nx.degree_centrality(G_cg, backend="cugraph")
        degree_cen = nx.degree_centrality(G)
        end = time.time()
        print(f'complete to calculate degree centrality {end - start:.5f} sec')

        start = time.time()
        with open(degree_cen_file, 'w') as f:
            json.dump(degree_cen, f)  # JSON 형식으로 저장
        end = time.time()
        print(f"Saved degree centrality in {end - start:.5f} sec")

    """
    df.apply 로직으로 degree 컬럼 추가 시 오류 발생
    - numpy.core._exceptions._ArrayMemoryError: 
    - Unable to allocate 50.4 GiB for an array with shape (22, 307596577) and data type object
    """
    # df['source_deg'] = df.apply(extract_value_from_dict, axis=1,
    #                            args=(degree_cen, 'source'))
    # df['target_deg'] = df.apply(extract_value_from_dict, axis=1,
    #                            args=(degree_cen, 'target'))

    # # Process finished with exit code 137
    # df['source_deg'] = df['source'].map(degree_cen).fillna(0)
    # df['target_deg'] = df['target'].map(degree_cen).fillna(0)

    # # Process finished with exit code 137
    # ddf = dd.from_pandas(df, npartitions=50)  # Dask로 변환 (50개 청크로 분할)
    # ddf['source_deg'] = ddf['source'].map(degree_cen).fillna(0)
    # ddf['target_deg'] = ddf['target'].map(degree_cen).fillna(0)
    # df = ddf.compute()  # 다시 pandas DataFrame으로 변환
